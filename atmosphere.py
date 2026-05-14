from dataclasses import dataclass
import numpy as np
from typing import Dict, Any
from constants import *
from atoms import MultiLevelAtom
from chemeq import compute_background_eos, get_partition_functions, partition_function

def get_angular_quadrature_1D(n_gauss):
    """
    Computes the angular quadrature, returning weights and coordinates separately.

    Args:
        n_gauss (int): The number of points for the Gauss-Legendre quadrature
                     (inclination angles).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - weights: A 1D array of quadrature weights of shape (n_gauss,).
            - points: A 1D array of inclinations in radians. Inclinations range from 0 to pi
    """
    # If the n_gauss is odd, add a point to avoid horizontal rays
    if n_gauss % 2 == 1:
        print(f"Warning: n_gauss={n_gauss} is odd, adding a ray to avoid horizontal ray.")
        n_gauss += 1

    mu, w_gauss = np.polynomial.legendre.leggauss(n_gauss)

    return w_gauss, mu

@dataclass
class Atmosphere:
    """
    Represents the physical conditions at different points in a stellar atmosphere.
    """
    zgrid: np.ndarray
    temp: np.ndarray
    pg: np.ndarray
    pel: np.ndarray

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atmosphere":
        """
        Factory method to create an Atmosphere instance from a dictionary.
        
        This method handles the conversion of standard Python lists (as loaded 
        from JSON) into NumPy arrays required by the dataclass.
        """
        return cls(
            zgrid=np.array(data['zgrid'])*1e5,
            temp=np.array(data['temp']),
            pg=np.array(data['pg']),
            pel=np.array(data['pel']),
        )

    @property
    def Ndepth(self) -> int:
        """Returns the number of depth points in the atmosphere."""
        return len(self.zgrid)

    def __post_init__(self):
        """
        Validates arrays and calculates rigorous background EOS.
        """
        if not (len(self.zgrid) == len(self.temp) == len(self.pg)):
            raise ValueError("All atmospheric arrays must have the same length.")
            
        # check that z is strictly increasing
        # (avoid weirdos that start the atmosphere from the top, please seek help if you want to do that)
        if not np.all(np.diff(self.zgrid) > 0):
            raise ValueError("zgrid must be strictly increasing.")
            
        # Instead of relying on the config's 'pel', we force the 92-element calculation
        print("Calculating baseline EOS using 92 elements...")
        calculated_ne, calculated_nh = compute_background_eos(self.temp, self.pg)
        
        self.ne = calculated_ne.copy() # self.pel/(self.temp*kB_CGS) 
        self.ne_bg = calculated_ne.copy() # self.pel/(self.temp*kB_CGS)
        self.nh = calculated_nh.copy() # self.pg/(self.temp*kB_CGS) - self.ne

def compute_lte_populations(atom: MultiLevelAtom, atmosphere: Atmosphere) -> np.ndarray:
    """
    Calculates LTE level populations for a multi-level atom using the Saha-Boltzmann 
    equations and Irwin partition functions to account for un-modeled higher states.

    Args:
        atom (MultiLevelAtom): The atomic model object.
        atmosphere (AtmosphereParams): A class containing atmospheric data.

    Returns:
        np.ndarray: A 2D array of shape (num_points, num_levels) containing
                    the population of each level at each atmospheric point.
    """

    levels = atom.levels
    num_levels = len(levels)
    energies = np.array([l.energy for l in levels])     # In ergs
    gs = np.array([l.g for l in levels])                # Statistical weights
    stages = np.array([l.ionization for l in levels])   # Ionization stage (e.g., 1 for neutral, 2 for singly ionized)

    temperature = atmosphere.temp
    electron_density = atmosphere.ne_bg
    h_density = atmosphere.nh
    num_points = len(temperature)

    # The Saha constant is (2 * pi * m_e * k_B / h^2)^1.5
    saha_const = ((2.0 * np.pi * m_e_CGS * kB_CGS) / (h_CGS**2))**1.5

    # Initialize the output array for populations
    populations = np.zeros((num_points, num_levels))
    
    # Find unique ionization stages and their ground state energies
    stages_unique = np.unique(stages)
    E_ground = {}
    for s in stages_unique:
        # The ground state energy of a stage is the minimum energy among its provided levels
        E_ground[s] = np.min(energies[stages == s])
        
    s_ref = np.min(stages_unique)
    E_ref = E_ground[s_ref]
    
    # Loop through each point in the atmosphere
    for k in range(num_points):
        T = temperature[k]
        if T <= 0.0:
            print(f"WARNING: Negative Temperature in lte populations at iz={k}. Skipping.")
            continue

        ne = electron_density[k]
        if ne <= 0.0:
            print(f"WARNING: Negative Electron Density in lte populations at iz={k}. Skipping.")
            continue
            
        kT = kB_CGS * T
        
        # Total number density for this element at depth k
        N_total = atom.abundance * h_density[k]
        
        # Saha factor: (2 / N_e) * (2 * pi * m_e * k_B * T / h^2)^1.5
        Phi = (2.0 / ne) * saha_const * (T ** 1.5)
        
        # Calculate Irwin partition functions for each stage at temperature T
        U_t = {}
        for s in stages_unique:
            U_t[s] = partition_function(atom, s, T)
            
        # Calculate the fractional abundance of each stage relative to the lowest provided stage (s_ref)
        f = {}
        for s in stages_unique:
            dE = E_ground[s] - E_ref
            # N_s / N_ref = (U_t / U_t_ref) * Phi^(s - s_ref) * exp(-dE / kT)
            f[s] = (U_t[s] / U_t[s_ref]) * (Phi ** (s - s_ref)) * np.exp(-dE / kT)
            
        sum_f = sum(f.values())
        
        # Total absolute population of the reference stage
        N_ref = N_total / sum_f
        
        # Total absolute population of each stage
        N_stage = {s: N_ref * f[s] for s in stages_unique}
        
        # Calculate the LTE population for each individual level (Boltzmann equation)
        for i in range(num_levels):
            s = stages[i]
            # Level energy relative strictly to its own stage's ground state
            E_level_rel = energies[i] - E_ground[s]
            
            # n_i = N_stage * (g_i / U_t) * exp(-E_rel / kT)
            populations[k, i] = N_stage[s] * (gs[i] / U_t[s]) * np.exp(-E_level_rel / kT)
        
    return populations
