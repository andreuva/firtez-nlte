import json, os, time, shutil
from glob import glob
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from constants import *
from atmosphere import Atmosphere, get_angular_quadrature_1D, compute_lte_populations
from atoms import MultiLevelAtom, create_frequency_grid
from formal_solver import plank

config_file = 'tests/freq_grid/config_lw.json'
# load the json configuration
with open(config_file) as f:
    configuration = json.load(f)
print(f"Configuration loaded from {config_file}.")

# Add a timestamp or unique identifier to the output directory to avoid overwriting previous runs
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
        print(f'ray {ir}\t weigth={weigths[ir]}\t inclination={np.rad2deg(np.arccos(ray))} deg')
    print('-'*25 + '\n')

print("Initializing atmosphere...")
atmosphere = Atmosphere.from_dict(configuration["atmosphere"])

atoms = [MultiLevelAtom.from_dict(config_atom) for config_atom in configuration["atoms"]]
for atom in atoms:
    atom.populations = compute_lte_populations(atom, atmosphere)
    atom.lte_populations = atom.populations.copy()
    atom.Js = np.zeros((atmosphere.Ndepth, len(atom.lines)))
    atom.compute_doppler_widths(atmosphere, configuration["atmosphere"]["turbulent_velocity"])

print("Creating frequency grid...\n")
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

print("Interpolating photoionization cross-sections...\n")
for atom in atoms:
    atom.photoionization_alphas = np.zeros((len(atom.continua), len(frequency_grid)))
    atom.lte_ratios_photoionization = np.zeros((atmosphere.Ndepth, len(atom.continua)))
    
    for i_cont, cont in enumerate(atom.continua):
        atom.lte_ratios_photoionization[:, i_cont] = (atom.lte_populations[:, cont.lower_level_index]/
                                                      np.maximum(atom.lte_populations[:, cont.upper_level_index], 1e-100))

        # Data from config is (wavelength [nm], sigma [cm^2])
        # Convert to (frequency [Hz], sigma [cm^2])
        cont_wls_cm = np.array([wl_nm*1e-7 for wl_nm, sig in cont.photoionization_cross_section])
        cont_alphas = np.array([sig for wl_nm, sig in cont.photoionization_cross_section])
        cont_nus = c_CGS / np.maximum(cont_wls_cm, 1e-300) # Avoid div by zero

        # Sort by increasing frequency
        sort_idx = np.argsort(cont_nus)
        cont_nus_sorted = cont_nus[sort_idx]
        cont_alphas_sorted = cont_alphas[sort_idx]
        
        # Interpolate onto the global grid, setting sigma=0 outside the continuum's range
        alphas_grid = np.interp(frequency_grid, cont_nus_sorted, cont_alphas_sorted, left=0.0, right=0.0)
        atom.photoionization_alphas[i_cont, :] = alphas_grid

# LAMBDA ITTERATIONS
for itteration in range(configuration["max_itterations"]):

    print('\n'+'--'*50)
    print(f'Itteration {itteration+1}/{configuration["max_itterations"]}')
    print('--'*50 + '\n')

    # Reset for the new iteration
    for atom in atoms:
        atom.Js = np.zeros((atmosphere.Ndepth, len(atom.lines)))
        atom.Lambda_star_bar = np.zeros((atmosphere.Ndepth, len(atom.lines))) # NEW
        atom.photoionization_rates = np.zeros((atmosphere.Ndepth, len(atom.continua)))
        atom.recombination_rates = np.zeros((atmosphere.Ndepth, len(atom.continua)))

    for ir, ray in tqdm(enumerate(rays), leave=False, desc="solving the RT to integrate Js"):

        # INITIAL CONDITIONS OF THE LONG CHARACTERISTICS RAY
        # check if the ray is downwards
        if ray > 0:
            iz_start, iz_end, step = 0, atmosphere.Ndepth, 1
            dz = atmosphere.zgrid[iz] - atmosphere.zgrid[iz - step]
            I_o = plank(frequency_grid, atmosphere.temp[0])
        else:
            iz_start, iz_end, step = atmosphere.Ndepth-1, -1, -1
            dz = atmosphere.zgrid[iz - step] - atmosphere.zgrid[iz]
            I_o = np.zeros_like(frequency_grid)
        
        # emis_m, abs_m = get_RT_coefficients(iz_start, frequency_grid, weigths_freq_grid, atoms, atmosphere)
        
        # Solve RT along the ray to integrate Js and other quantities.
        for iz in range(iz_start + step, iz_end, step):

            I_m = I_o.copy()
            # emis_o, abs_o = get_RT_coefficients(iz, frequency_grid, weigths_freq_grid, atoms, atmosphere)
            # I_o = compute_RT_solver(ray, I_m, dz, emis_m, emis_o, abs_m, abs_o)
            # I_o, Lambda_star_mu = compute_RT_solver(ray, I_m, dz, emis_m, emis_o, abs_m, abs_o)


            # emis_m, abs_m = emis_o.copy(), abs_o.copy()

    max_relative_change = 0.0 #solve_SEE(atoms, atmosphere)
    if max_relative_change < configuration["max_tolerance"]:
        print("NLTE converged!")
        break
    print(f"Iteration {itteration+1} with a max relative change of: {max_relative_change}")
    print("-"*50 + "\n")

