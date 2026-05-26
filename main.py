import json, os, time, shutil
from glob import glob
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from constants import *
from atmosphere import Atmosphere, get_angular_quadrature_1D, compute_lte_populations
from atoms import MultiLevelAtom, create_frequency_grid, solve_SEE
from formal_solver import plank, voigt, formal_solution, get_RT_coefficients

config_file = 'config_lw.json'
# load the json configuration
with open(config_file) as f:
    configuration = json.load(f)
print(f"Configuration loaded from {config_file}.")

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
if eval(configuration["debug"]):
    print('-'*25)
    for ir, ray in enumerate(rays):
        print(f'ray {ir}\t weigth={weigths[ir]}\t mu={ray}\t inclination={np.rad2deg(np.arccos(ray))} deg')
    print('-'*25 + '\n')

print("Initializing atmosphere...")
atmosphere = Atmosphere.from_dict(configuration["atmosphere"])

print("Loading atomic models...")
atoms = [MultiLevelAtom.from_dict(config_atom) for config_atom in configuration["atoms"]]
for atom in atoms:
    atom.populations = compute_lte_populations(atom, atmosphere)
    atom.lte_populations = atom.populations.copy()
    atom.Js = np.zeros((atmosphere.Ndepth, len(atom.lines)))
    atom.compute_doppler_widths(atmosphere, configuration["atmosphere"]["turbulent_velocity"])

print("Creating frequency grid...")
frequency_grid, weigths_freq_grid = create_frequency_grid(atoms,
                                                        #   temperature=5700,
                                                          v_turb=configuration["atmosphere"]["turbulent_velocity"],
                                                        #   max_resol_nm=configuration["max_wavelength_resolution_nm"],
                                                        #   min_resol_nm=configuration["min_wavelength_resolution_nm"]
                                                          )

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
# Pre-calculate line broadening parameters for each line and depth point
for atom in atoms:
    atom_mass_CGS = atom.mass * m_u_CGS
    for line in atom.lines:
        # Defaults for depth-dependent terms
        line.vdw_cross = 0.0
        line.stark_vrel_factor = 0.0
        line.stark_c23 = 0.0
        line.lin_stark_factor = 0.0
        
        upper_lvl = atom.levels[line.upper_level_index]
        lower_lvl = atom.levels[line.lower_level_index]
        current_ion = upper_lvl.ionization
        
        # Get overarching continuum level for limits
        cont_level = next((lvl for lvl in atom.levels if lvl.ionization == current_ion + 1), None)
        E_cont = cont_level.energy if cont_level else atom.levels[-1].energy

        for elastic in line.broadening.elastic:
            if elastic.get("type") == "VdwUnsold":
                '''
                Implementation of the Unsold method for van der Waals broadening.
                Follows LW and HM2014 pp. 237-238,
                '''
                vals = elastic.get("vals", [1.0, 1.0])
                deltaR = (E_Ryd_erg / (E_cont - upper_lvl.energy))**2 - (E_Ryd_erg / (E_cont - lower_lvl.energy))**2
                Z = upper_lvl.ionization + 1
                
                C6_CGS = 2.5 * q_e_CGS**2 * alpha_H_CGS * 2.0 * np.pi * (Z * a0_CGS)**2 / h_CGS * abs(deltaR)
                C625 = C6_CGS**0.4
                
                vRel35H = (8.0 * kB_CGS / (np.pi * atom_mass_CGS) * (1.0 + atom_mass_CGS / m_H_CGS))**0.3
                vRel35He = (8.0 * kB_CGS / (np.pi * atom_mass_CGS) * (1.0 + atom_mass_CGS / m_He_CGS))**0.3
                
                line.vdw_cross = 8.08 * (vals[0] * vRel35H + vals[1] * atmosphere.he_abund * vRel35He) * C625
                
            elif elastic.get("type") == "QuadraticStarkBroadening":
                '''
                Lindholm theory result for Quadratic Stark broadening by electrons and
                singly ionised particles.
                Follows HM2014 pp. 238-239, uses C4 from Traving 1960 via LW (and previously RH).
                '''
                coeff = elastic.get("coeff", 1.0)
                C_stark = 8.0 * kB_CGS / (np.pi * atom_mass_CGS)
                # 28.0 is average atomic weight
                Cm = (1.0 + atom_mass_CGS / m_e_CGS)**(1.0/6.0) \
                    + (1.0 + atom_mass_CGS / (28.0 * m_u_CGS))**(1.0/6.0)
                line.stark_vrel_factor = (C_stark)**(1.0/6.0) * Cm
                
                E_Ryd_elem = E_Ryd_erg / (1.0 + m_e_CGS / atom_mass_CGS)
                Z_i = lower_lvl.ionization + 1
                neff_l = Z_i * np.sqrt(E_Ryd_elem / (E_cont - lower_lvl.energy))
                neff_u = Z_i * np.sqrt(E_Ryd_elem / (E_cont - upper_lvl.energy))
                
                C4 = q_e_CGS**2 \
                   * a0_CGS \
                   * (2.0 * np.pi * a0_CGS**2 / h_CGS) / (18.0 * Z_i**4) * \
                     abs((neff_u * (5.0 * neff_u**2 + 1.0))**2 \
                         - (neff_l * (5.0 * neff_l**2 + 1.0))**2)
                line.stark_c23 = 11.37 * (coeff * C4)**(2.0/3.0)
                
            elif elastic.get("type") == "HydrogenLinearStarkBroadening":
                """ 
                Linear Stark broadening for the case of Hydrogen from Sutton 1978 (like LW and RH).
                """     
                nUpper = np.round(np.sqrt(0.5 * upper_lvl.g))
                nLower = np.round(np.sqrt(0.5 * lower_lvl.g))
                a1 = 0.642 if nUpper - nLower == 1 else 1.0
                cc = a1 * 0.6 * (nUpper**2 - nLower**2)
                # Lightweaver's unit (cm-2 translation drops the 10^-4 scalar when taking ne_CGS vs ne_SI)
                line.lin_stark_factor = cc * 4.0 * np.pi * 0.425
            
            else:
                raise NotImplementedError(f"Elastic broadening type {elastic.get('type')} not implemented.")

# #################################################################################
# LAMBDA ITTERATIONS
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

    for ir, ray in tqdm(enumerate(rays), total=len(rays), leave=False, desc="solving the RT to integrate Js"):

        # INITIAL CONDITIONS OF THE LONG CHARACTERISTICS RAY
        if ray > 0:
            downward_ray = False
            iz_start, iz_end, step = 0, atmosphere.Ndepth, 1
            I_o = plank(frequency_grid, atmosphere.temp[0])
        else:
            downward_ray = True
            iz_start, iz_end, step = atmosphere.Ndepth-1, -1, -1
            I_o = np.zeros_like(frequency_grid)
        
        # Solve RT along the ray to integrate Js and other quantities.
        for iz in range(iz_start, iz_end, step):

            if iz == iz_start:
                # Boundary point: I_o is already set. Just get local RT coeffs for the integrals.
                emis_O, abs_O = get_RT_coefficients(iz, frequency_grid, weigths_freq_grid, atoms, atmosphere)
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
                emis_M, abs_M = emis_O.copy(), abs_O.copy()

                emis_O, abs_O = get_RT_coefficients(iz, frequency_grid, weigths_freq_grid, atoms, atmosphere)
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

                    total_damping = 0.0
                    dop_freq = (frequency_grid - line.nu0)/atom.doppler_widths[iz, il]

                    for natural in line.broadening.natural:
                        if natural.get("type") == "RadiativeBroadening":
                            total_damping += natural.get("gamma", 0.0)
                        else:
                            raise NotImplementedError(f"Natural broadening type {natural.get('type')} not implemented.")

                    for elastic in line.broadening.elastic:
                        if elastic.get("type") == "VdwUnsold":
                            total_damping += line.vdw_cross * atmosphere.temp[iz]**0.3 * nHGround
                        elif elastic.get("type") == "QuadraticStarkBroadening":
                            total_damping += line.stark_c23 * line.stark_vrel_factor * atmosphere.temp[iz]**(1.0/6.0) * atmosphere.ne[iz]
                        elif elastic.get("type") == "HydrogenLinearStarkBroadening":
                            total_damping += line.lin_stark_factor * atmosphere.ne[iz]**(2.0/3.0)
                        else:
                            raise NotImplementedError(f"Elastic broadening type {elastic.get('type')} not implemented.")

                    a_damp = total_damping / (4 * np.pi * atom.doppler_widths[iz, il])
                    voigt_line = voigt(dop_freq, a_damp).real

                    # Truncate/zero the profile outside the line's own physical grid boundary
                    mask = np.abs(frequency_grid - line.nu0) <= line.max_delta_nu
                    voigt_line[~mask] = 0.0

                    voigt_norm = voigt_line / np.sum(voigt_line*weigths_freq_grid)

                    # # --- Calculate line opacity and ratio for lambda_star ---
                    nu_pop = atom.populations[iz, line.upper_level_index]
                    nl_pop = atom.populations[iz, line.lower_level_index]

                    abs_line = h_CGS * line.nu0 / (4 * np.pi) * (nl_pop * line.Blu - nu_pop * line.Bul) * voigt_norm
                    opacity_ratio = np.zeros_like(abs_O)
                    valid_abs = abs_O > 1e-100
                    opacity_ratio[valid_abs] = abs_line[valid_abs] / abs_O[valid_abs]
                    # opacity_ratio = np.clip(opacity_ratio, 0.0, 10.0)

                    # --- Integrate the J and the lambda_star ---
                    atom.Js[iz, il] += 0.5 * np.sum(weigths[ir]*weigths_freq_grid*I_o*voigt_norm)
                    # Integrate the Lambda operator
                    atom.Lambda_star_bar[iz, il] += 0.5 * np.sum(weigths[ir] * weigths_freq_grid * Lambda_star_mu * opacity_ratio * voigt_norm)

            J_grid[iz, :] += 0.5 * weigths[ir] * I_o

        if ray == np.max(rays):
            emergent_I_vertical = I_o.copy()
    
    atmosphere.J_nu = J_grid

    # Calculate Photoionization/Recombination rates OUTSIDE the ray loop
    for iz in range(atmosphere.Ndepth):
        for atom in atoms:
            for i_cont, cont in enumerate(atom.continua):
                alphas = atom.photoionization_alphas[i_cont, :]
                active_idx = alphas > 0
                if not np.any(active_idx): continue
                
                # Spontaneous + Stimulated (using mean intensity)
                stim_spont_term = (2.0 * hnu3_grid[active_idx] / c_CGS**2 + J_grid[iz, active_idx]) \
                                * np.exp(-hnu_grid[active_idx] / (kB_CGS * atmosphere.temp[iz]))
                                
                integrand_ki = (alphas[active_idx] / hnu_grid[active_idx]) * atom.lte_ratios_photoionization[iz, i_cont] \
                               * stim_spont_term
                integrand_ik = (alphas[active_idx] / hnu_grid[active_idx]) * J_grid[iz, active_idx]

                R_ik = 4.0 * np.pi * np.sum(weigths_freq_grid[active_idx] * integrand_ik)
                R_ki = 4.0 * np.pi * np.sum(weigths_freq_grid[active_idx] * integrand_ki)
                
                atom.photoionization_rates[iz, i_cont] = R_ik
                atom.recombination_rates[iz, i_cont] = R_ki

    max_relative_change = solve_SEE(atoms, atmosphere)
    print(f"Iteration {itteration+1} with a max relative change of: {max_relative_change}")
    for atom in atoms:
        print(f"  {atom.name} max Lambda_star_bar: {np.max(atom.Lambda_star_bar)}")
    print("-"*50 + "\n")

    # =========================================================================
    # ITERATION-BY-ITERATION DEBUG PLOTS
    # =========================================================================
    if configuration.get("save_dir", False):
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
        if 'emergent_I_vertical' in locals():
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

    if max_relative_change < configuration["max_tolerance"]:
        print("NLTE converged!")
        break
# #################################################################################

# =============================================================================
# FINAL FORMAL SOLUTION — compute the emergent spectrum with converged populations
# for a single vertical ray (μ = 1, θ = 0) and save all results.
# =============================================================================
print("\n" + "==" * 50)
print("Computing final formal solution with converged populations (μ=1)...")
print("==" * 50 + "\n")

ray_mu1 = 1.0   # vertically outgoing ray
# Upward ray: start from the bottom (deepest point), propagate upward
tau_depth = np.zeros((atmosphere.Ndepth, len(frequency_grid)))
source_func = np.zeros_like(tau_depth)
plank_func = np.zeros_like(tau_depth)

I_o = plank(frequency_grid, atmosphere.temp[0])
emis_O, abs_O = get_RT_coefficients(0, frequency_grid, weigths_freq_grid, atoms, atmosphere)
tau_depth[0,:] = abs_O * (atmosphere.zgrid[1] - atmosphere.zgrid[0]) / ray_mu1  # zero, but for consistency
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
wavelength_grid_nm_final = (c_CGS / frequency_grid) * 1e7  # cm to nm

# ---- Save results to disk ----
if configuration.get("save_dir", False):
    print(f"Saving results to {configuration["save_dir"]}...")

    # Save the frequency and wavelength grids
    np.save(os.path.join(configuration["save_dir"], "frequency_grid_hz.npy"), frequency_grid)
    np.save(os.path.join(configuration["save_dir"], "wavelength_grid_nm.npy"), wavelength_grid_nm_final)

    # Save emergent intensity at mu=1 [n_freq]
    np.save(os.path.join(configuration["save_dir"], "emergent_intensity_mu1.npy"), I_disk_centre)

    # Save optical depth arrays from the vertical (μ=1) formal solution
    # tau_depth : shape (Ndepth, Nfreq)  – cumulative τ from bottom up to each layer
    # tau_surface: shape (Nfreq,)        – total column optical depth
    np.save(os.path.join(configuration["save_dir"], "optical_depth_vs_depth_mu1.npy"), tau_depth)
    np.save(os.path.join(configuration["save_dir"], "optical_depth_total_mu1.npy"),   tau_surface)

    # Save converged populations, LTE populations, and Js for each atom
    for atom in atoms:
        np.save(os.path.join(configuration["save_dir"], f"populations_{atom.name}.npy"), atom.populations)
        np.save(os.path.join(configuration["save_dir"], f"lte_populations_{atom.name}.npy"), atom.lte_populations)
        np.save(os.path.join(configuration["save_dir"], f"Js_{atom.name}.npy"), atom.Js)

if configuration.get("debug", False):
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
            tau_sel = np.flip(tau_depth[:, mask_nu], axis=1)  # shape (Ndepth, Nsel)

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

    print("Done! All results saved.")
