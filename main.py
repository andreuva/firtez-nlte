import json, os, sys, time, shutil
from glob import glob
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from constants import *
from atmosphere import Atmosphere, get_angular_quadrature_1D, compute_lte_populations
from atoms import MultiLevelAtom, create_frequency_grid, solve_SEE
from atoms import (compute_line_frequency_weights, continuum_frequency_weights,
                   init_line_broadening, VMICRO_FLOOR)
from atoms import validate_ionization_stages, reconcile_abundances
from atoms import get_barklem_cross_section
from formal_solver import plank, voigt, formal_solution, get_RT_coefficients, line_profile

# Config path may be given as argv[1]; otherwise fall back to the historical default.
# Without this the SNAPI comparison harness silently ran the default config instead of the
# one it generated (audit F-013).
config_file = sys.argv[1] if len(sys.argv) > 1 else 'config_H_Ca_Mg_Na.json'
# load the json configuration
with open(config_file) as f:
    configuration = json.load(f)
print(f"Configuration loaded from {config_file}.")


def _as_bool(value, default=False):
    """
    Interpret a JSON config flag as a boolean.

    'debug' has historically been written as the *string* "True"/"False". A bare
    truthiness test makes "False" true, which is why the two debug gates in this file used
    to disagree (eval() at the top, .get() at the bottom). Handle both spellings once
    (audit F-009).
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if value is None:
        return default
    return bool(value)


DEBUG = _as_bool(configuration.get("debug", False))
# Electron density treatment. See solve_SEE's docstring for the three modes:
#   "eos"   hold n_e at the background LTE equation-of-state value
#   "delta" n_e = n_e,bg + delta-charge of the active atoms; charge conservation, but
#           anchored to the background, whose donors stay frozen at n_e,bg
#   "nlte"  full charge conservation: active atoms donate from their NLTE populations and
#           every other element is re-evaluated from Saha at the current n_e
ELECTRON_MODE = str(configuration.get("electron_density_mode", "nlte")).lower()
if ELECTRON_MODE not in ("eos", "delta", "nlte"):
    raise ValueError(f"electron_density_mode must be 'eos', 'delta' or 'nlte'; got {ELECTRON_MODE!r}")
print(f"Electron density mode: {ELECTRON_MODE}"
      + {"eos":   "  (held at the background EOS value)",
         "delta": "  (n_e,bg + active-atom delta-charge; passive donors frozen)",
         "nlte":  "  (full charge conservation; passive donors re-evaluated at the current n_e)"}[ELECTRON_MODE])

# Add a timestamp or unique identifier to the output directory to avoid overwriting previous runs
if configuration.get("save_dir", False):
    if configuration.get("new_dir", True):
        configuration["save_dir"] = f"outputs/{configuration['save_dir']}_{time.strftime('%Y_%m_%d_%H%M%S', time.localtime())}"

    print(f"Output will be saved to: {configuration['save_dir']}")
    if not os.path.exists(configuration["save_dir"]):
        os.makedirs(configuration["save_dir"])

    # copy the config file and all the executable files to the output directory for reference
    with open(os.path.join(configuration["save_dir"], os.path.basename(config_file)), 'w') as f_out:
        json.dump(configuration, f_out, indent=2)
    for file in glob('*.py'):
        shutil.copy2(file, os.path.join(configuration["save_dir"], file))
    print("Configuration and code files copied to output directory for reference.\n")

print("Starting NLTE model run...\n")

weigths, rays = get_angular_quadrature_1D(configuration["quadrature"]["n_gaus"])
print("Quadrature initialized.")
if DEBUG:
    print('-'*25)
    for ir, ray in enumerate(rays):
        print(f'ray {ir}\t weigth={weigths[ir]}\t mu={ray}\t inclination={np.rad2deg(np.arccos(ray))} deg')
    print('-'*25 + '\n')

print("Initializing atmosphere...")
atmosphere = Atmosphere.from_dict(configuration["atmosphere"])

# turbulent_velocity may be a scalar or a per-depth list (FAL-C's v_turb varies 1.8-6.8 km/s)
turbulent_velocity = np.asarray(configuration["atmosphere"]["turbulent_velocity"], dtype=float)
if turbulent_velocity.ndim == 0:
    turbulent_velocity = float(turbulent_velocity)

print("Loading atomic models...")
atoms = [MultiLevelAtom.from_dict(config_atom) for config_atom in configuration["atoms"]]
for _a in atoms:
    print(f"  {_a.name}: {'ACTIVE (NLTE)' if _a.is_active else 'PASSIVE (held at LTE)'}")
validate_ionization_stages(atoms)
reconcile_abundances(atoms)
for atom in atoms:
    atom.populations = compute_lte_populations(atom, atmosphere)
    atom.lte_populations = atom.populations.copy()
    # Model-completeness check. The SNAPI-style conservation row retains this explicit LTE
    # population instead of filling the model levels with the whole element. The remainder
    # stays in the LTE EOS reservoir and has no explicit NLTE transitions.
    _frac = atom.lte_populations.sum(axis=1) / (atom.abundance * atmosphere.nh)
    if _frac.min() < 0.9:
        _k = int(np.argmin(_frac))
        print(f"  WARNING: {atom.name}: LTE populations account for only "
              f"{100*_frac.min():.2f}% of the element at T={atmosphere.temp[_k]:.0f} K "
              f"(z={atmosphere.zgrid[_k]/1e5:.0f} km). The model atom is incomplete there -- "
              f"missing ionization stages or unmodelled excited levels remain in the LTE "
              f"EOS reservoir and cannot participate in explicit NLTE transitions.")
    atom.Js = np.zeros((atmosphere.Ndepth, len(atom.lines)))
    atom.compute_doppler_widths(atmosphere, turbulent_velocity)

print("Creating frequency grid...")
frequency_grid, weigths_freq_grid = create_frequency_grid(atoms,
                                                          atmosphere=atmosphere,
                                                          v_turb=turbulent_velocity,
                                                        #   max_resol_nm=configuration["max_wavelength_resolution_nm"],
                                                        #   min_resol_nm=configuration["min_wavelength_resolution_nm"]
                                                          )

# Per-line quadrature weights confined to each line's own window (audit F-002).
compute_line_frequency_weights(atoms, frequency_grid)

atmosphere.J_nu = np.zeros((atmosphere.Ndepth, len(frequency_grid)))
for iz, TT in enumerate(atmosphere.temp):
    atmosphere.J_nu[iz, :] = plank(frequency_grid, TT)

# Pre-calculate h*nu for the grid
hnu_grid = h_CGS * frequency_grid
hnu_grid[hnu_grid == 0] = 1e-100 # Avoid division by zero
hnu3_grid = h_CGS * frequency_grid**3

# --------------------------
print("Interpolating photoionization cross-sections...\n")

# Use a wavelength grid derived straight from our active frequencies to interpolate our alphas
wavelength_grid_nm = (c_CGS / frequency_grid) * 1e7

for atom in atoms:
    atom.photoionization_alphas = np.zeros((len(atom.continua), len(frequency_grid)))
    atom.lte_ratios_photoionization = np.zeros((atmosphere.Ndepth, len(atom.continua)))
    
    for i_cont, cont in enumerate(atom.continua):
        atom.lte_ratios_photoionization[:, i_cont] = (atom.lte_populations[:, cont.lower_level_index]/
                                                      np.maximum(atom.lte_populations[:, cont.upper_level_index], 1e-100))

        # Direct evaluation of analytical Hydrogenic arrays or dynamically interpolated explicit boundaries
        alphas_grid = cont.alpha(wavelength_grid_nm, atom.levels)
        atom.photoionization_alphas[i_cont, :] = alphas_grid

# --------------------------
# Pre-calculate the depth-independent line broadening constants (audit F-009: this used to
# be an inline block whose side effects get_RT_coefficients silently depended on).
init_line_broadening(atoms, atmosphere)

def bottom_boundary_intensity(freq_grid, freq_weights, atoms, atmosphere, mu):
    """
    Lower boundary condition: the diffusion approximation
        I+ = B_nu(T) + mu * dB_nu/dtau_nu
    as used by RH, Lightweaver and SNAPI, rather than a plain I+ = B_nu.

    dB/dtau is evaluated from the two deepest points with the local total opacity, so the
    correction vanishes smoothly when the bottom is optically thick.

    For the FAL-C models in use the correction is immaterial -- the minimum total optical
    depth over the whole frequency grid is 20.2, so the boundary term is attenuated by
    e^-20 ~ 2e-9 (audit F-008). It is implemented because the plain-B choice is badly
    wrong for optically thinner models, not because it fixes an observed discrepancy.
    """
    B0 = plank(freq_grid, atmosphere.temp[0])
    B1 = plank(freq_grid, atmosphere.temp[1])
    _, chi0 = get_RT_coefficients(0, freq_grid, freq_weights, atoms, atmosphere)
    _, chi1 = get_RT_coefficients(1, freq_grid, freq_weights, atoms, atmosphere)
    dz = atmosphere.zgrid[1] - atmosphere.zgrid[0]
    dtau = 0.5 * (chi0 + chi1) * abs(dz)
    dBdtau = np.where(dtau > 0.0, (B1 - B0) / np.where(dtau > 0.0, dtau, 1.0), 0.0)
    # tau increases downward (into the star); z increases upward, hence the sign.
    return B0 - mu * dBdtau


# #################################################################################
# LAMBDA ITTERATIONS
converged = False
prev_change = 0.0
rho = 0.0
solver_failed = False
max_relative_change = np.inf
for itteration in range(configuration["max_itterations"]):

    print('--'*50)
    print(f'Itteration {itteration+1}/{configuration["max_itterations"]}')
    print('--'*50 + '\n')

    # Reset for the new iteration
    for atom in atoms:
        atom.Js = np.zeros((atmosphere.Ndepth, len(atom.lines)))
        atom.Lambda_star_bar = np.zeros((atmosphere.Ndepth, len(atom.lines))) # NEW
        atom.photoionization_rates = np.zeros((atmosphere.Ndepth, len(atom.continua)))
        atom.recombination_rates = np.zeros((atmosphere.Ndepth, len(atom.continua)))
    
    J_grid = np.zeros((atmosphere.Ndepth, len(frequency_grid)))
    emergent_I_vertical = None

    # Opacities and emissivities depend only on depth, NOT on the ray, but they used to be
    # recomputed inside the ray loop -- an Nrays-fold repetition of the most expensive
    # kernel in the code (the scalar Peach/Fe I loops in the background opacity). Cache them
    # once per iteration: identical numbers, ~Nrays times faster (audit F-009).
    emis_cache = np.zeros((atmosphere.Ndepth, len(frequency_grid)))
    abs_cache = np.zeros((atmosphere.Ndepth, len(frequency_grid)))
    for iz in tqdm(range(atmosphere.Ndepth), leave=False, desc="RT coefficients"):
        emis_cache[iz, :], abs_cache[iz, :] = get_RT_coefficients(
            iz, frequency_grid, weigths_freq_grid, atoms, atmosphere)

    for ir, ray in tqdm(enumerate(rays), total=len(rays), leave=False, desc="solving the RT to integrate Js"):

        # INITIAL CONDITIONS OF THE LONG CHARACTERISTICS RAY
        if ray > 0:
            downward_ray = False
            iz_start, iz_end, step = 0, atmosphere.Ndepth, 1
            I_o = bottom_boundary_intensity(frequency_grid, weigths_freq_grid, atoms, atmosphere, ray)
        else:
            downward_ray = True
            iz_start, iz_end, step = atmosphere.Ndepth-1, -1, -1
            I_o = np.zeros_like(frequency_grid)
        
        # Solve RT along the ray to integrate Js and other quantities.
        for iz in range(iz_start, iz_end, step):

            if iz == iz_start:
                # Boundary point: I_o is already set. Just get local RT coeffs for the integrals.
                emis_O, abs_O = emis_cache[iz, :], abs_cache[iz, :]
                # Boundary intensity is prescribed, so it doesn't depend on local source function
                Lambda_star_mu = np.zeros_like(frequency_grid) 
            else:
                # Inner points: Propagate the formal solution as normal
                if downward_ray:
                    dz = atmosphere.zgrid[iz - step] - atmosphere.zgrid[iz]
                else:
                    dz = atmosphere.zgrid[iz] - atmosphere.zgrid[iz - step]

                # move the O point to M
                I_m = I_o.copy()
                emis_M, abs_M = emis_O, abs_O

                emis_O, abs_O = emis_cache[iz, :], abs_cache[iz, :]
                I_o, Lambda_star_mu = formal_solution(ray, I_m, dz, emis_M, emis_O, abs_M, abs_O)

            # --- Now the integration applies to ALL points, including the boundary ---
            h_atom = next((a for a in atoms if a.name == "H"), None)
            # True ground state hydrogen mapping. Falls back to background total H if not existing in config.
            nHGround = h_atom.populations[iz, 0] if h_atom else atmosphere.nh[iz]

            # go trhough all the active atoms
            for atom in atoms:
                
                # -------------------
                # go through all the lines of the atom to compute Js
                # that will be used to compute the bound-bound Radiative rates.
                for il, line in enumerate(atom.lines):

                    # Identical profile to the one get_RT_coefficients uses -- shared helper,
                    # so the two can no longer drift apart (audit F-009).
                    voigt_norm = line_profile(line, atom, iz, frequency_grid, atmosphere, nHGround)

                    # # --- Calculate line opacity and ratio for lambda_star ---
                    nu_pop = atom.populations[iz, line.upper_level_index]
                    nl_pop = atom.populations[iz, line.lower_level_index]

                    abs_line = h_CGS * line.nu0 / (4 * np.pi) * (nl_pop * line.Blu - nu_pop * line.Bul) * voigt_norm
                    opacity_ratio = np.zeros_like(abs_O)
                    valid_abs = abs_O > 1e-100
                    opacity_ratio[valid_abs] = abs_line[valid_abs] / abs_O[valid_abs]
                    # opacity_ratio = np.clip(opacity_ratio, 0.0, 10.0)

                    # --- Integrate the J and the lambda_star ---
                    atom.Js[iz, il] += 0.5 * np.sum(weigths[ir]*line.freq_weights*I_o*voigt_norm)
                    # Integrate the Lambda operator
                    atom.Lambda_star_bar[iz, il] += 0.5 * np.sum(weigths[ir] * line.freq_weights * Lambda_star_mu * opacity_ratio * voigt_norm)

            J_grid[iz, :] += 0.5 * weigths[ir] * I_o

        if ray == np.max(rays):
            emergent_I_vertical = I_o.copy()
    
    atmosphere.J_nu = J_grid

    # Calculate Photoionization/Recombination rates OUTSIDE the ray loop
    for iz in range(atmosphere.Ndepth):
        for atom in atoms:
            for i_cont, cont in enumerate(atom.continua):
                alphas = atom.photoionization_alphas[i_cont, :]
                active_idx, continuum_weights = continuum_frequency_weights(
                    alphas, frequency_grid, weigths_freq_grid)
                if active_idx.size == 0: continue
                
                # Spontaneous + Stimulated (using mean intensity)
                stim_spont_term = (2.0 * hnu3_grid[active_idx] / c_CGS**2 + J_grid[iz, active_idx]) \
                                  * np.exp(-hnu_grid[active_idx] / (kB_CGS * atmosphere.temp[iz]))
                                
                integrand_ki = (alphas[active_idx] / hnu_grid[active_idx]) * atom.lte_ratios_photoionization[iz, i_cont] \
                               * stim_spont_term
                integrand_ik = (alphas[active_idx] / hnu_grid[active_idx]) * J_grid[iz, active_idx]

                R_ik = 4.0 * np.pi * np.sum(continuum_weights * integrand_ik)
                R_ki = 4.0 * np.pi * np.sum(continuum_weights * integrand_ki)
                
                atom.photoionization_rates[iz, i_cont] = R_ik
                atom.recombination_rates[iz, i_cont] = R_ki

    max_relative_change, see_ok = solve_SEE(
        atoms, atmosphere, electron_mode=ELECTRON_MODE, return_status=True)
    print(f"Iteration {itteration+1} with a max relative change of: {max_relative_change}")
    if not see_ok:
        solver_failed = True
        print("NLTE statistical-equilibrium solve failed; using LTE fallback.")
        break
    for atom in atoms:
        print(f"  {atom.name} max Lambda_star_bar: {np.max(atom.Lambda_star_bar)}")
    print("-"*50 + "\n")

    # =========================================================================
    # ITERATION-BY-ITERATION DEBUG PLOTS
    # =========================================================================
    if configuration.get("save_dir", False) and DEBUG:
        wavelength_nm_plot = (c_CGS / frequency_grid) * 1e7

        # 1. Population Plot (NLTE vs LTE)
        for atom in atoms:
            plt.figure(figsize=(10, 6), dpi=100)
            for i in range(atom.populations.shape[-1]):
                plt.plot(atmosphere.zgrid / 1e5, atom.populations[:, i], '-', color=f'C{i}',
                         label=f"NLTE Level {i}")
                plt.plot(atmosphere.zgrid / 1e5, atom.lte_populations[:, i], 'o', color=f'C{i}',
                         alpha=0.4, label=f"LTE Level {i}")
            plt.xlabel("Height (km)")
            plt.ylabel("Population (cm$^{-3}$)")
            plt.yscale("log")
            plt.title(f"Iteration {itteration+1} populations — {atom.name}")
            plt.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4)
            plt.tight_layout()
            plt.savefig(os.path.join(configuration["save_dir"], f"debug_populations_{atom.name}_iter_{itteration+1}.png"))
            plt.close()

        # 2. Line Profile (Emergent Intensity for the most vertical ray)
        if emergent_I_vertical is not None:
            for atom in atoms:
                for il, line in enumerate(atom.lines):
                    line_wl_nm = line.lambda0 * 1e7
                    wl_window = 1.0  # nm
                    mask_wl = (wavelength_nm_plot > line_wl_nm - wl_window) & (wavelength_nm_plot < line_wl_nm + wl_window)
                    if not np.any(mask_wl): continue
                    
                    plt.figure(figsize=(8, 5), dpi=100)
                    plt.plot(wavelength_nm_plot[mask_wl], emergent_I_vertical[mask_wl], 'k-', linewidth=1.5)
                    plt.axvline(line_wl_nm, color='red', linestyle=':', alpha=0.5, label=f"Line Center: {line_wl_nm:.2f} nm")
                    plt.xlabel("Wavelength (nm)")
                    plt.ylabel("Intensity (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)")
                    plt.title(f"Emergent Line Profile (most vertical ray) — Iter {itteration+1} — {atom.name} {line_wl_nm:.2f} nm")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(configuration["save_dir"], f"debug_profile_{atom.name}_line{il}_iter_{itteration+1}.png"))
                    plt.close()

        # 3. Radiation Field J vs B comparison at key depths
        depth_indices = [0, atmosphere.Ndepth // 2, atmosphere.Ndepth - 1]
        depth_labels = ["Photosphere (Bottom)", "Mid-Chromosphere", "Top of Atmosphere"]
        
        for atom in atoms:
            for il, line in enumerate(atom.lines):
                line_wl_nm = line.lambda0 * 1e7
                wl_window = 1.0  # nm
                mask_wl = (wavelength_nm_plot > line_wl_nm - wl_window) & (wavelength_nm_plot < line_wl_nm + wl_window)
                if not np.any(mask_wl): continue
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
                for idx, iz_plot in enumerate(depth_indices):
                    ax = axes[idx]
                    B_vals = plank(frequency_grid[mask_wl], atmosphere.temp[iz_plot])
                    J_vals = J_grid[iz_plot, mask_wl]
                    
                    ax.plot(wavelength_nm_plot[mask_wl], J_vals, 'b-', label=r'$J_\nu$ (Mean Radiation)')
                    ax.plot(wavelength_nm_plot[mask_wl], B_vals, 'r--', label=r'$B_\nu$ (Planck Function)')
                    ax.set_xlabel("Wavelength (nm)")
                    ax.set_ylabel("Intensity")
                    ax.set_title(f"{depth_labels[idx]} (H={atmosphere.zgrid[iz_plot]/1e5:.0f} km)")
                    ax.legend(fontsize=8)
                plt.suptitle(f"J vs B — Iteration {itteration+1} — {atom.name} {line_wl_nm:.2f} nm", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(configuration["save_dir"], f"debug_J_vs_B_{atom.name}_line{il}_iter_{itteration+1}.png"))
                plt.close()

        # 4. Statistical Equilibrium Rates Plot (C vs R) vs Height
        for atom in atoms:
            if not hasattr(atom, 'R_matrix_all'): continue
            Nlevel = len(atom.levels)
            for il, line in enumerate(atom.lines):
                i = line.lower_level_index
                j = line.upper_level_index
                
                plt.figure(figsize=(10, 6), dpi=100)
                n_i = atom.populations[:, i]
                n_j = atom.populations[:, j]
                
                rate_R_up = np.maximum(n_i * atom.R_matrix_all[:, i, j], 1e-30)
                rate_R_down = np.maximum(n_j * atom.R_matrix_all[:, j, i], 1e-30)
                rate_C_up = np.maximum(n_i * atom.C_matrix_all[:, i, j], 1e-30)
                rate_C_down = np.maximum(n_j * atom.C_matrix_all[:, j, i], 1e-30)
                
                plt.plot(atmosphere.zgrid / 1e5, rate_R_up, 'b-', label=f"Radiative Up ($n_{i} \\times R_{{ij}}$)")
                plt.plot(atmosphere.zgrid / 1e5, rate_R_down, 'b--', label=f"Radiative Down ($n_{j} \\times R_{{ji}}$)")
                plt.plot(atmosphere.zgrid / 1e5, rate_C_up, 'r-', label=f"Collisional Up ($n_{i} \\times C_{{ij}}$)")
                plt.plot(atmosphere.zgrid / 1e5, rate_C_down, 'r--', label=f"Collisional Down ($n_{j} \\times C_{{ji}}$)")
                
                plt.xlabel("Height (km)")
                plt.ylabel("Transition Rate (cm$^{-3}$ s$^{-1}$)")
                plt.yscale("log")
                plt.title(f"SEE Transition Rates (Level {i} <-> {j}) — Iter {itteration+1} — {atom.name}")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(configuration["save_dir"], f"debug_rates_{atom.name}_trans_{i}_{j}_iter_{itteration+1}.png"))
                plt.close()

    # Stop on the ESTIMATED REMAINING ERROR, not on the size of the last step.
    #
    # MALI with a diagonal approximate operator converges geometrically, and on the
    # reference problem the measured ratio is rho = 0.961: the per-iteration change and the
    # distance to the true solution differ by rho/(1 - rho) = 25. Stopping when the change
    # first drops below 1e-3 left the departure coefficients 2.6e-2 from the converged
    # answer -- 26x the tolerance the run reports -- worst on the hydrogen n = 2, 3 and 4
    # populations in the chromosphere, which is exactly what H-alpha is built from.
    #
    # sum_{m>0} rho^m * delta = delta * rho / (1 - rho), so require that, not delta. The
    # extra min with the tolerance keeps the test from ever stopping EARLIER than the old
    # one when rho < 0.5, so a noisy ratio can only delay convergence, never fake it.
    max_tol = configuration["max_tolerance"]
    if prev_change > 0.0 and max_relative_change > 0.0:
        rho = max_relative_change / prev_change
        eff_tol = max_tol * min(1.0, (1.0 - rho) / max(rho, 1e-30)) if rho < 1.0 else -1.0
    else:
        eff_tol = max_tol
    prev_change = max_relative_change

    if max_relative_change < eff_tol:
        converged = True
        print(f"NLTE converged!  (convergence ratio rho = {rho:.4f}, estimated remaining "
              f"error = {max_relative_change * rho / (1.0 - rho):.4E})")
        break
# #################################################################################

if not converged:
    atmosphere.ne[:] = atmosphere.ne_bg
    for atom in atoms:
        atom.populations[:] = atom.lte_populations
    reason = "solver failure" if solver_failed else "iteration limit"
    print(f"NLTE did not converge ({reason}); populations and electron density reset to LTE.")

# =============================================================================
# DEPARTURE COEFFICIENTS  β_i = n_i(NLTE) / n_i(LTE)
# =============================================================================
for atom in atoms:
    atom.departure_coefficients = atom.populations / np.maximum(atom.lte_populations, 1e-100)
    print(f"{atom.name} departure coefficients (min/max per level):")
    for i in range(atom.populations.shape[-1]):
        print(f"  Level {i}: beta_min={atom.departure_coefficients[:, i].min():.4f}  "
              f"beta_max={atom.departure_coefficients[:, i].max():.4f}")
print()

# =============================================================================
# FINAL FORMAL SOLUTION — compute the emergent spectrum with converged populations
# for a single vertical ray (μ = 1, θ = 0) and save all results.
# =============================================================================
print("\n" + "==" * 50)
print("Computing final formal solution with converged populations (μ=1)...")
print("==" * 50 + "\n")

# Emergent intensity for every outgoing quadrature ray, plus the vertical mu = 1 ray.
# The Lambda loop only ever kept the mu = 1 spectrum; the centre-to-limb variation is a
# direct observable and costs one extra formal solution per ray.
emergent_rays = {}
_mu_out = sorted([m for m in rays if m > 0])
for _mu in _mu_out:
    _I = bottom_boundary_intensity(frequency_grid, weigths_freq_grid, atoms, atmosphere, _mu)
    _eO, _aO = get_RT_coefficients(0, frequency_grid, weigths_freq_grid, atoms, atmosphere)
    for _iz in range(1, atmosphere.Ndepth):
        _dz = atmosphere.zgrid[_iz] - atmosphere.zgrid[_iz - 1]
        _eM, _aM = _eO, _aO
        _eO, _aO = get_RT_coefficients(_iz, frequency_grid, weigths_freq_grid, atoms, atmosphere)
        _I, _ = formal_solution(_mu, _I, _dz, _eM, _eO, _aM, _aO)
    emergent_rays[_mu] = _I.copy()
    print(f"  emergent intensity computed for mu = {_mu:.4f} "
          f"(theta = {np.rad2deg(np.arccos(_mu)):.1f} deg)")

ray_mu1 = 1.0   # vertically outgoing ray
# Upward ray: start from the bottom (deepest point), propagate upward
tau_depth = np.zeros((atmosphere.Ndepth, len(frequency_grid)))
source_func = np.zeros_like(tau_depth)
plank_func = np.zeros_like(tau_depth)

I_o = bottom_boundary_intensity(frequency_grid, weigths_freq_grid, atoms, atmosphere, ray_mu1)
emis_O, abs_O = get_RT_coefficients(0, frequency_grid, weigths_freq_grid, atoms, atmosphere)
tau_depth[0,:] = 0.0  # tau is measured from the bottom, so the bottom row is zero (F-009)
source_func[0,:] = emis_O / np.maximum(abs_O, 1e-100)
plank_func[0,:] = plank(frequency_grid, atmosphere.temp[0])
emis_depth = np.zeros_like(tau_depth)
abs_depth = np.zeros_like(tau_depth)
emis_depth[0,:] = emis_O
abs_depth[0,:] = abs_O

for iz in range(1, atmosphere.Ndepth):
    dz = atmosphere.zgrid[iz] - atmosphere.zgrid[iz - 1]   # always positive (upward)

    I_m = I_o.copy()
    emis_M, abs_M = emis_O.copy(), abs_O.copy()
    
    emis_O, abs_O = get_RT_coefficients(iz, frequency_grid, weigths_freq_grid, atoms, atmosphere)
    delta_tau = 0.5 * (abs_M + abs_O) * np.abs(dz / ray_mu1)
    tau_depth[iz, :] = tau_depth[iz - 1, :] + delta_tau
    source_func[iz, :] = emis_O / np.maximum(abs_O, 1e-100)
    plank_func[iz, :] = plank(frequency_grid, atmosphere.temp[iz])
    emis_depth[iz, :] = emis_O
    abs_depth[iz, :] = abs_O

    I_o, _ = formal_solution(ray_mu1, I_m, dz, emis_M, emis_O, abs_M, abs_O)

# I_o now contains the emergent intensity at μ=1
I_disk_centre = I_o
# tau_depth[iz, :] == optical depth from the bottom up to layer iz.
# The surface value tau_depth[-1, :] is the total optical depth of the whole atmosphere.
tau_surface = np.log10(tau_depth[-1, :])
# convert optical depth to a typical scale where 0 is the observer and max is the bottom of the atmosphere
tau_depth_observed = tau_depth[-1, :][np.newaxis, :] - tau_depth
wavelength_grid_nm_final = (c_CGS / frequency_grid) * 1e7  # cm to nm

# ---- Save results to disk ----
if configuration.get("save_dir", False):
    print(f"Saving results to {configuration["save_dir"]}...")

    # Save the frequency and wavelength grids
    np.save(os.path.join(configuration["save_dir"], "frequency_grid_hz.npy"), frequency_grid)
    np.save(os.path.join(configuration["save_dir"], "wavelength_grid_nm.npy"), wavelength_grid_nm_final)

    # Save emergent intensity at mu=1 [n_freq]
    np.save(os.path.join(configuration["save_dir"], "emergent_intensity_mu1.npy"), I_disk_centre)
    np.save(os.path.join(configuration["save_dir"], "emergent_intensity_mu_values.npy"),
            np.array(_mu_out))
    np.save(os.path.join(configuration["save_dir"], "emergent_intensity_rays.npy"),
            np.array([emergent_rays[m] for m in _mu_out]))

    # Save optical depth arrays from the vertical (μ=1) formal solution
    # tau_depth : shape (Ndepth, Nfreq)  – cumulative τ from bottom up to each layer
    # tau_surface: shape (Nfreq,)        – total column optical depth
    np.save(os.path.join(configuration["save_dir"], "optical_depth_vs_depth_mu1.npy"), tau_depth_observed)
    np.save(os.path.join(configuration["save_dir"], "optical_depth_total_mu1.npy"),   tau_surface)

    # The converged electron density and the rate matrices are needed to diagnose the
    # ionization balance against a reference code; without them a post-hoc analysis has to
    # fall back on ne_bg and silently misattributes the difference (audit/BENCHMARK.md).
    np.save(os.path.join(configuration["save_dir"], "electron_density.npy"), atmosphere.ne)
    np.save(os.path.join(configuration["save_dir"], "electron_density_bg.npy"), atmosphere.ne_bg)

    # Save converged populations, LTE populations, and Js for each atom
    for atom in atoms:
        np.save(os.path.join(configuration["save_dir"], f"populations_{atom.name}.npy"), atom.populations)
        np.save(os.path.join(configuration["save_dir"], f"lte_populations_{atom.name}.npy"), atom.lte_populations)
        np.save(os.path.join(configuration["save_dir"], f"Js_{atom.name}.npy"), atom.Js)
        np.save(os.path.join(configuration["save_dir"], f"departure_coefficients_{atom.name}.npy"),
                atom.departure_coefficients)
        np.save(os.path.join(configuration["save_dir"], f"photoionization_rates_{atom.name}.npy"),
                atom.photoionization_rates)
        np.save(os.path.join(configuration["save_dir"], f"recombination_rates_{atom.name}.npy"),
                atom.recombination_rates)
        if hasattr(atom, "R_matrix_all"):
            np.save(os.path.join(configuration["save_dir"], f"R_matrix_{atom.name}.npy"), atom.R_matrix_all)
            np.save(os.path.join(configuration["save_dir"], f"C_matrix_{atom.name}.npy"), atom.C_matrix_all)

if True:
    # Disk-centre emergent spectrum (μ=1 ray)
    wavelength_nm_plot = np.flip(wavelength_grid_nm_final)
    I_plot = np.flip(I_disk_centre)

    # Global spectrum overview
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(wavelength_nm_plot, I_plot, 'k-', linewidth=0.5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)")
    plt.title("Emergent Spectrum (disk centre, μ=1)")
    plt.tight_layout()
    plt.savefig(os.path.join(configuration["save_dir"], "emergent_spectrum_overview.png"))
    plt.close()

    # ---- Emergent intensity for every outgoing ray (centre-to-limb variation) ----
    plt.figure(figsize=(12, 6), dpi=110)
    _cm = plt.cm.viridis(np.linspace(0, 0.9, len(_mu_out)))
    for _i, _mu in enumerate(_mu_out):
        plt.plot(wavelength_nm_plot, np.flip(emergent_rays[_mu]), '-', lw=0.6, color=_cm[_i],
                 label=f"$\\mu$={_mu:.3f} ({np.rad2deg(np.arccos(_mu)):.0f}°)")
    plt.plot(wavelength_nm_plot, I_plot, 'k--', lw=0.8, label=r"$\mu$=1 (disk centre)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)")
    plt.title("Emergent intensity per ray")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(configuration["save_dir"], "emergent_intensity_rays.png"))
    plt.close()

    # Per-line centre-to-limb: profile for every ray, one panel per line
    for iat, atom in enumerate(atoms):
        for il, line in enumerate(atom.lines):
            line_wl_nm = line.lambda0 * 1e7
            m_wl = (wavelength_nm_plot > line_wl_nm - 0.5) & (wavelength_nm_plot < line_wl_nm + 0.5)
            if not np.any(m_wl):
                continue
            fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=110)
            for _i, _mu in enumerate(_mu_out):
                _Ir = np.flip(emergent_rays[_mu])
                ax[0].plot(wavelength_nm_plot[m_wl], _Ir[m_wl], '-', color=_cm[_i], lw=1.2,
                           label=f"$\\mu$={_mu:.3f}")
                ax[1].plot(wavelength_nm_plot[m_wl], _Ir[m_wl] / np.max(_Ir[m_wl]), '-',
                           color=_cm[_i], lw=1.2)
            ax[0].plot(wavelength_nm_plot[m_wl], I_plot[m_wl], 'k--', lw=1.0, label=r"$\mu$=1")
            ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Intensity")
            ax[0].set_title(f"{atom.name} {line_wl_nm:.3f} nm — centre to limb")
            ax[0].legend(fontsize=7); ax[0].grid(ls=':', alpha=.5)
            ax[1].set_xlabel("Wavelength (nm)"); ax[1].set_ylabel("I / max(I)")
            ax[1].set_title("normalised (line shape vs $\\mu$)")
            ax[1].grid(ls=':', alpha=.5)
            plt.tight_layout()
            plt.savefig(os.path.join(configuration["save_dir"],
                                     f"emergent_rays_{atom.name}_line{il}.png"))
            plt.close()

    # Zoom into each line
    for iat, atom in enumerate(atoms):
        for il, line in enumerate(atom.lines):
            line_wl_nm = line.lambda0 * 1e7  # cm -> nm
            wl_window = 1.0  # nm half-width for zoom
            mask_wl = (wavelength_nm_plot > line_wl_nm - wl_window) & (wavelength_nm_plot < line_wl_nm + wl_window)

            if not np.any(mask_wl):
                continue

            plt.figure(figsize=(8, 5), dpi=100)
            plt.plot(wavelength_nm_plot[mask_wl], I_plot[mask_wl], 'k-', linewidth=1)
            plt.axvline(line_wl_nm, color='red', linestyle=':', alpha=0.5, label=f"$\\lambda_0$ = {line_wl_nm:.2f} nm")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)")
            plt.title(f"Emergent Line Profile — {atom.name}, {line_wl_nm:.2f} nm")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(configuration["save_dir"], f"emergent_line_{atom.name}_line{il}.png"))
            plt.close()

    # ---- Optical depth diagnostic plots ----
    # Total optical depth vs wavelength
    wl_nm_plot_od = np.flip(wavelength_grid_nm_final)
    tau_surf_plot  = np.flip(tau_surface)

    plt.figure(figsize=(12, 5), dpi=100)
    plt.semilogy(wl_nm_plot_od, tau_surf_plot, 'b-', linewidth=0.7)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"$log_{10}(\tau_\nu)$")
    plt.tight_layout()
    plt.savefig(os.path.join(configuration["save_dir"], "optical_depth_total_spectrum.png"))
    # plt.show()
    plt.close()

    # Optical depth as a function of height and wavelength – one panel per spectral line
    height_km = atmosphere.zgrid / 1e5
    for iat, atom in enumerate(atoms):
        for il, line in enumerate(atom.lines):
            line_wl_nm = line.lambda0 * 1e7
            wl_window  = 1.0  # nm
            # Indices in the *original* (frequency-ordered) frequency grid
            mask_nu = ((wavelength_grid_nm_final > line_wl_nm - wl_window) &
                       (wavelength_grid_nm_final < line_wl_nm + wl_window))
            if not np.any(mask_nu):
                continue

            wl_sel = np.flip(wavelength_grid_nm_final[mask_nu])
            tau_sel = np.flip(tau_depth_observed[:, mask_nu], axis=1)  # shape (Ndepth, Nsel)

            plt.figure(figsize=(14, 5), dpi=100)
            plt.pcolormesh(wl_sel, np.arange(len(height_km)),
                               np.log10(np.maximum(tau_sel, 1e-10)),
                               cmap='plasma', shading='auto')
            plt.yticks(np.arange(0, len(height_km), 7), [f"{h:.0f}" for h in height_km[::7]])
            plt.colorbar(label=r'$\log_{10}(\tau_\nu)$')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Height (km)")
            plt.title(f"Optical depth map — {atom.name} {line_wl_nm:.2f} nm")

            plt.tight_layout()
            plt.savefig(os.path.join(configuration["save_dir"],
                                     f"optical_depth_{atom.name}_line{il}.png"))
            # plt.show()
            plt.close()

    # Source function, Planck function, and Ratio diagnostics
    for iat, atom in enumerate(atoms):
        for il, line in enumerate(atom.lines):
            line_wl_nm = line.lambda0 * 1e7
            wl_window  = 1.0  # nm
            # Indices in the *original* (frequency-ordered) frequency grid
            mask_nu = ((wavelength_grid_nm_final > line_wl_nm - wl_window) &
                       (wavelength_grid_nm_final < line_wl_nm + wl_window))
            if not np.any(mask_nu):
                continue

            wl_sel = np.flip(wavelength_grid_nm_final[mask_nu])
            S_sel = np.flip(source_func[:, mask_nu], axis=1)
            B_sel = np.flip(plank_func[:, mask_nu], axis=1)
            
            # S/B ratio
            ratio_sel = S_sel / np.maximum(B_sel, 1e-100)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
            
            # 1. Source Function Map
            im1 = axes[0].pcolormesh(wl_sel, np.arange(len(height_km)),
                                     np.log10(np.maximum(S_sel, 1e-100)),
                                     cmap='plasma', shading='auto')
            axes[0].set_yticks(np.arange(0, len(height_km), 7))
            axes[0].set_yticklabels([f"{h:.0f}" for h in height_km[::7]])
            axes[0].set_xlabel("Wavelength (nm)")
            axes[0].set_ylabel("Height (km)")
            axes[0].set_title(f"Source Function $\\log_{{10}}(S_\\nu)$")
            fig.colorbar(im1, ax=axes[0], label=r'$\log_{10}(S_\nu)$')
            vmin, vmax = im1.get_clim()

            # 2. Planck Function Map
            im2 = axes[1].pcolormesh(wl_sel, np.arange(len(height_km)),
                                     np.log10(np.maximum(B_sel, 1e-100)),
                                     cmap='plasma', shading='auto',
                                     vmin=vmin, vmax=vmax)
            axes[1].set_yticks(np.arange(0, len(height_km), 7))
            axes[1].set_yticklabels([f"{h:.0f}" for h in height_km[::7]])
            axes[1].set_xlabel("Wavelength (nm)")
            axes[1].set_ylabel("Height (km)")
            axes[1].set_title(f"Planck Function $\\log_{{10}}(B_\\nu)$")
            fig.colorbar(im2, ax=axes[1], label=r'$\log_{10}(B_\nu)$')

            # 3. Ratio S/B Map
            im3 = axes[2].pcolormesh(wl_sel, np.arange(len(height_km)),
                                     ratio_sel,
                                     cmap='coolwarm', shading='auto', vmin=0, vmax=2)
            axes[2].set_yticks(np.arange(0, len(height_km), 7))
            axes[2].set_yticklabels([f"{h:.0f}" for h in height_km[::7]])
            axes[2].set_xlabel("Wavelength (nm)")
            axes[2].set_ylabel("Height (km)")
            axes[2].set_title(f"Ratio $S_\\nu / B_\\nu$")
            fig.colorbar(im3, ax=axes[2], label=r'$S_\nu / B_\nu$')

            plt.suptitle(f"S/B Diagnostics — {atom.name} {line_wl_nm:.2f} nm", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(configuration["save_dir"],
                                     f"SB_diagnostics_{atom.name}_line{il}.png"))
            # plt.show()
            plt.close()

    # Final converged population profiles vs height
    for iat, atom in enumerate(atoms):
        plt.figure(figsize=(10, 7.5), dpi=100)
        for i in range(atom.populations.shape[-1]):
            plt.plot(atmosphere.zgrid / 1e5, atom.populations[:, i], '-', color=f'C{i}',
                    label=f"NLTE Level {i}")
            plt.plot(atmosphere.zgrid / 1e5, atom.lte_populations[:, i], 'o', color=f'C{i}',
                    alpha=0.5, label=f"LTE  Level {i}")
        plt.xlabel("Height (km)")
        plt.ylabel("Population (cm$^{-3}$)")
        plt.yscale("log")
        # plt.xscale("log")
        plt.title(f"Converged Populations — {atom.name}")
        plt.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, edgecolor='none', framealpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(configuration["save_dir"], f"final_populations_{atom.name}.png"))
        plt.close()

    # Departure coefficients β_i = n_NLTE / n_LTE  vs height
    for iat, atom in enumerate(atoms):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        for i in range(atom.departure_coefficients.shape[-1]):
            ax.plot(atmosphere.zgrid / 1e5,
                    atom.departure_coefficients[:, i],
                    '-', color=f'C{i}', label=f"Level {i}")
        ax.axhline(1.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6, label="LTE (β=1)")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e2)
        ax.set_xlabel("Height (km)")
        ax.set_ylabel(r"Departure coefficient $\beta_i = n_i^{\rm NLTE} / n_i^{\rm LTE}$")
        ax.set_title(f"Departure Coefficients — {atom.name}")
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.12),
                  ncol=min(atom.departure_coefficients.shape[-1] + 1, 6),
                  edgecolor='none', framealpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(configuration["save_dir"], f"departure_coefficients_{atom.name}.png"))
        plt.close(fig)

    print("Done! All results saved.")
