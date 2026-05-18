import json, os, time, shutil
from glob import glob
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from constants import *
from atmosphere import Atmosphere, get_angular_quadrature_1D, compute_lte_populations
from atoms import MultiLevelAtom, create_frequency_grid
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
        
        emis_O, abs_O = get_RT_coefficients(iz_start, frequency_grid, weigths_freq_grid, atoms, atmosphere)
        
        # Solve RT along the ray to integrate Js and other quantities.
        for iz in range(iz_start + step, iz_end, step):

            # compute the geometrical path length dz for the current step (not necesarily constant)
            if downward_ray:
                dz = atmosphere.zgrid[iz - step] - atmosphere.zgrid[iz]
            else:
                dz = atmosphere.zgrid[iz] - atmosphere.zgrid[iz - step]

            # move the O point to M
            I_m = I_o.copy()
            emis_M, abs_M = emis_O.copy(), abs_O.copy()

            # compute the RT coeffs. in O
            emis_O, abs_O = get_RT_coefficients(iz, frequency_grid, weigths_freq_grid, atoms, atmosphere)
            # Compute the outgoing intentensity at point O, and the MALI contribution Lambda_star_mu at point O.
            I_o, _ = formal_solution(ray, I_m, dz, emis_M, emis_O, abs_M, abs_O)

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
                    voigt_norm = voigt_line / np.sum(voigt_line*weigths_freq_grid)

                    # Sum the intensity over the wavelength
                    atom.Js[iz, il] += np.sum(weigths[ir]*weigths_freq_grid*I_o*voigt_norm)
                    # # Integrate the Lambda operator
                    # atom.Lambda_star_bar[iz, il] += np.sum(weigths[ir] * weigths_freq_grid * Lambda_star_mu * opacity_ratio * voigt_norm)

                # go through all the continua of the atom
                # compute the photoionization and recombination rates
                # this will then be used to compute the bound-free Radiative rates for the SEE.
                for i_cont, cont in enumerate(atom.continua):
                    alphas = atom.photoionization_alphas[i_cont, :]
                    
                    # Restrict to non-zero continuum wavelengths
                    active_idx = alphas > 0
                    if not np.any(active_idx):
                        continue
                        
                    # Integration: alpha_v / h_v * I_v * d_v * dOmega
                    integrand_ik = (alphas[active_idx] / hnu_grid[active_idx]) * I_o[active_idx]
                    
                    stim_spont_term = (2.0 * hnu3_grid[active_idx] / c_CGS**2 + I_o[active_idx]) \
                                    * np.exp(-hnu_grid[active_idx] / (kB_CGS * atmosphere.temp[iz]))
                    integrand_ki = (alphas[active_idx] / hnu_grid[active_idx]) * atom.lte_ratios_photoionization[iz, i_cont] \
                                   * stim_spont_term
                    
                    # Rates scaled by radiation Solid angle integral equivalences (2*pi for 1D) 
                    R_ik = 2.0 * np.pi * np.sum(weigths[ir] * weigths_freq_grid[active_idx] * integrand_ik)
                    R_ki = 2.0 * np.pi * np.sum(weigths[ir] * weigths_freq_grid[active_idx] * integrand_ki)
                    
                    atom.photoionization_rates[iz, i_cont] += R_ik
                    atom.recombination_rates[iz, i_cont] += R_ki


    max_relative_change = 0.0 #solve_SEE(atoms, atmosphere)
    print(f"Iteration {itteration+1} with a max relative change of: {max_relative_change}")
    print("-"*50 + "\n")
    if max_relative_change < configuration["max_tolerance"]:
        print("NLTE converged!")
        break
# #################################################################################

