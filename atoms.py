from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
from constants import *

if TYPE_CHECKING:
    from atmosphere import Atmosphere

@dataclass
class Level:
    """Represents a single energy level of an atom."""
    energy: float
    g: float
    label: str
    ionization: int  # 0 for neutral, 1 for singly ionized, etc.
    J: float
    L: int
    S: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Level":
        return cls(**data)

@dataclass
class Quadrature:
    """Represents the wavelength integration/resolution parameters for a line."""
    N_lambda: int
    q_core: float
    q_wing: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Quadrature":
        return cls(**data)

@dataclass
class Broadening:
    """Represents the broadening mechanisms affecting a line."""
    natural: List[Dict[str, Any]]
    elastic: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Broadening":
        return cls(
            natural=data.get('natural', []),
            elastic=data.get('elastic', [])
        )

@dataclass
class Line:
    """Represents a bound-bound transition (spectral line)."""
    upper_level_index: int
    lower_level_index: int
    oscillator_strength: float
    type: str
    quadrature: Quadrature
    broadening: Broadening
    energy: float = field(init=False, default=0.0)
    nu0: float = field(init=False, default=0.0)
    lambda0: float = field(init=False, default=0.0)
    Aul: float = field(init=False, default=0.0)
    Bul: float = field(init=False, default=0.0)
    Blu: float = field(init=False, default=0.0)
    delta_E: float = field(init=False, default=0.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Line":
        # Create a shallow copy so we don't modify the original dict, 
        # then instantiate nested structures
        data_copy = data.copy()
        data_copy['quadrature'] = Quadrature.from_dict(data_copy.get('quadrature', {}))
        data_copy['broadening'] = Broadening.from_dict(data_copy.get('broadening', {}))
        return cls(**data_copy)

@dataclass
class Continuum:
    """Base class for bound-free transitions."""
    upper_level_index: int
    lower_level_index: int
    type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Continuum":
        ctype = data.get("type", "ExplicitContinuum")
        if ctype == "ExplicitContinuum":
            return ExplicitContinuum.from_dict(data)
        elif ctype == "HydrogenicContinuum":
            return HydrogenicContinuum.from_dict(data)
        else:
            raise ValueError(f"Unknown continuum type: {ctype}")

    def get_lambda_edge_nm(self, levels: List[Level]) -> float:
        """Returns the rest wavelength (ionization edge) in nm."""
        delta_E_erg = levels[self.upper_level_index].energy - levels[self.lower_level_index].energy
        return (h_CGS * c_CGS / delta_E_erg) * 1e7

    def get_wavelength_grid(self, levels: List[Level]) -> np.ndarray:
        raise NotImplementedError

    def alpha(self, wavelength_nm: np.ndarray, levels: List[Level]) -> np.ndarray:
        raise NotImplementedError

@dataclass
class ExplicitContinuum(Continuum):
    upper_level_index: int
    lower_level_index: int
    type: str
    photoionization_cross_section: List[Tuple[float, float]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplicitContinuum":
        # Convert explicit data from m^2 to cm^2
        pcs = [(float(w), float(c) * 1e4) for w, c in data.get('photoionization_cross_section', [])]
        return cls(
            upper_level_index=data['upper_level_index'],
            lower_level_index=data['lower_level_index'],
            type=data.get('type', 'ExplicitContinuum'),
            photoionization_cross_section=pcs
        )

    def get_wavelength_grid(self, levels: List[Level]) -> np.ndarray:
        grid = np.array([w for w, _ in self.photoionization_cross_section])
        edge_nm = self.get_lambda_edge_nm(levels)
        if edge_nm - grid[-1] > 0.1:
            grid = np.append(grid, edge_nm)
        elif grid[-1] > edge_nm:
            grid = grid[grid <= edge_nm]
            if len(grid) == 0 or edge_nm - grid[-1] > 0.01:
                grid = np.append(grid, edge_nm)
        return grid

    def alpha(self, wavelength_nm: np.ndarray, levels: List[Level]) -> np.ndarray:
        grid_nm = np.array([w for w, _ in self.photoionization_cross_section])
        grid_alpha = np.array([a for _, a in self.photoionization_cross_section])
        edge_nm = self.get_lambda_edge_nm(levels)
        min_nm = grid_nm[0]

        alpha_interp = np.interp(wavelength_nm, grid_nm, grid_alpha, left=0.0, right=0.0)
        alpha_interp[wavelength_nm < min_nm] = 0.0
        alpha_interp[wavelength_nm > edge_nm] = 0.0
        alpha_interp[alpha_interp < 0.0] = 0.0
        return alpha_interp

@dataclass
class HydrogenicContinuum(Continuum):
    upper_level_index: int
    lower_level_index: int
    type: str
    NlambdaGen: int
    alpha0: float
    minWavelength: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HydrogenicContinuum":
        return cls(
            upper_level_index=data['upper_level_index'],
            lower_level_index=data['lower_level_index'],
            type=data.get('type', 'HydrogenicContinuum'),
            NlambdaGen=data['NlambdaGen'],
            alpha0=data['alpha0'] * 1e4,  # Convert m^2 to cm^2
            minWavelength=data['minWavelength']
        )

    def get_wavelength_grid(self, levels: List[Level]) -> np.ndarray:
        edge_nm = self.get_lambda_edge_nm(levels)
        return np.linspace(self.minWavelength, edge_nm, self.NlambdaGen)

    def alpha(self, wavelength_nm: np.ndarray, levels: List[Level]) -> np.ndarray:
        edge_nm = self.get_lambda_edge_nm(levels)

        Z = levels[self.upper_level_index].ionization
        nEff = Z * np.sqrt( E_Ryd_erg / (levels[self.upper_level_index].energy - levels[self.lower_level_index].energy))

        gbf0 = gaunt_bf(edge_nm, nEff, Z)
        gbf = gaunt_bf(wavelength_nm, nEff, Z)

        alpha_vals = self.alpha0 * gbf / gbf0 * (wavelength_nm / edge_nm)**3
        alpha_vals[wavelength_nm < self.minWavelength] = 0.0
        alpha_vals[wavelength_nm > edge_nm] = 0.0
        return alpha_vals

def gaunt_bf(wvl, nEff, charge) -> float:
    '''
    Gaunt factor for bound-free transitions, from Seaton (1960), Rep. Prog.
    Phys. 23, 313, as used in RH. COPIED FROM LW

    Parameters
    ----------
    wvl : float or array-like
        The wavelength at which to compute the Gaunt factor [nm].
    nEff : float
        Principal quantum number.
    charge : float
        Charge of free state.

    Returns
    -------
    result : float or array-like
        Gaunt factor for bound-free transitions.
    '''
    # /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313 -- ----------- */
    # Copied from RH, ensuring vectorisation support
    x = h_CGS * c_CGS / (wvl * 1e-7) / (E_Ryd_erg * charge**2)
    x3 = x**(1.0/3.0)
    nsqx = 1.0 / (nEff**2 *x)

    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) - 0.0496 * x3**2 \
            * (1.0 - (1.0 - nsqx) * (2.0 / 3.0) * nsqx)
@dataclass
class Collision:
    """Represents a collisional transition between levels."""
    type: str  # 'E' for electron, 'H' for neutral hydrogen, etc.
    upper_level_index: int
    lower_level_index: int
    temperatures: List[float]
    rates: List[float]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Collision":
        return cls(**data)

@dataclass
class MultiLevelAtom:
    """Top-level class to represent a multi-level atom model."""
    name: str
    abundance: float
    mass: float
    Z: int
    levels: List[Level] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    continua: List[Continuum] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)
    irwin_coefficients: Dict[int, List[float]] = field(default_factory=dict)

    populations: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    lte_populations: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    Js: np.ndarray = field(init=False, default_factory=lambda: np.array([])) # Mean line-integrated intensities
    
    photoionization_rates: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    recombination_rates: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    photoionization_alphas: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    
    doppler_widths: np.ndarray = field(init=False, default_factory=lambda: np.array([]))

    def __post_init__(self):
        """
        Calculates and populates intrinsic line properties after the
        atom object is created.
        """
        # Convert all level energies from wavenumbers (cm^-1) to ergs.
        for level in self.levels:
            level.energy = level.energy * h_CGS * c_CGS

        # Note: This sorting breaks the indices provided in the atom file.
        # So we need to remap indices in transitions
        ilev_sorted = np.argsort([level.energy for level in self.levels])
        self.levels = [self.levels[i] for i in ilev_sorted]

        remap_idx = np.argsort(ilev_sorted)
        for line in self.lines:
            line.upper_level_index = remap_idx[line.upper_level_index]
            line.lower_level_index = remap_idx[line.lower_level_index]
        for cont in self.continua:
            cont.upper_level_index = remap_idx[cont.upper_level_index]
            cont.lower_level_index = remap_idx[cont.lower_level_index]
        for col in self.collisions:
            col.upper_level_index = remap_idx[col.upper_level_index]
            col.lower_level_index = remap_idx[col.lower_level_index]

        for line in self.lines:
            # Ensure an odd number of points for symmetry based on the new nested structure
            # if line.quadrature.N_lambda % 2 == 0:
            #     line.quadrature.N_lambda += 1


            upper_level = self.levels[line.upper_level_index]
            lower_level = self.levels[line.lower_level_index]

            if upper_level.energy > lower_level.energy:
                line.delta_E = upper_level.energy - lower_level.energy
                line.nu0 = line.delta_E / h_CGS
                # Calculate wavelength in cm
                line.lambda0 = (h_CGS * c_CGS / line.delta_E)
            else:
                # This should ideally raise an error, but we set to 0
                # to avoid crashes if data is malformed.
                print(f"WARNING: Line with upper level energy <= lower level energy in atom {self.name}. Setting line properties to zero.")
                line.delta_E, line.nu0, line.lambda0 = 0.0, 0.0, 0.0

            gl, gu = self.levels[line.lower_level_index].g, self.levels[line.upper_level_index].g

            line.Aul = 8*np.pi**2 *q_e_CGS**2 *line.nu0**2 / \
                        (c_CGS**3*m_e_CGS) * (gl/gu) * line.oscillator_strength
            line.Bul = c_CGS**2/(2*h_CGS*line.nu0**3) * line.Aul
            line.Blu = line.Bul * (gu/gl)

            # print(f"Initialized line {line.type} with nu0={line.nu0:.3e} Hz, lambda0={line.lambda0:.3e} cm, Aul={line.Aul:.3e}, Bul={line.Bul:.3e}, Blu={line.Blu:.3e} for atom {self.name}.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiLevelAtom":
        """Factory method to create a MultiLevelAtom from a dictionary."""
        return cls(
            name=data['name'],
            abundance=data['abundance'],
            mass=data['mass'],
            Z=data['Z'],
            levels=[Level.from_dict(l) for l in data.get('levels', [])],
            lines=[Line.from_dict(l) for l in data.get('lines', [])],
            continua=[Continuum.from_dict(c) for c in data.get('continua', [])],
            collisions=[Collision.from_dict(c) for c in data.get('collisions', [])],
            irwin_coefficients={int(k): v for k, v in data.get('irwin_coefficients', {}).items()}
        )

    def compute_doppler_widths(self, atmosphere: "Atmosphere", v_turb: float):
        """
        Pre-calculates the Doppler widths for all lines at all depths.
        Stores the result in self.doppler_widths [Ndepth, Nlines].
        """
        Ndepth = atmosphere.Ndepth
        Nlines = len(self.lines)
        self.doppler_widths = np.zeros((Ndepth, Nlines))
        
        temp = atmosphere.temp
        if len(temp) != Ndepth:
            raise ValueError("Atmosphere temperature array length mismatch.")

        # Thermal velocity component: sqrt(2kT/m)
        v_thermal = np.sqrt(2.0 * kB_CGS * temp / (m_u_CGS * self.mass))
        v_doppler = np.sqrt(v_thermal**2 + v_turb**2)

        for il, line in enumerate(self.lines):
            if line.nu0 > 0:
                self.doppler_widths[:, il] = (line.nu0 / c_CGS) * v_doppler

def create_frequency_grid(atoms: List[MultiLevelAtom], 
                        #   temperature: float,
                        #   v_turb: float = 0.0,
                        #   max_resol_nm: float = 3e-4,
                        #   min_resol_nm: float = 1e1,) -> Tuple[np.ndarray, np.ndarray]:
                          v_turb: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a global frequency grid with trapezoidal integration weights.

    The dimensionless frequency grid for each line is generated using an
    analytical formula that provides a smooth transition from a densely
    sampled core to a sparsely sampled wing.

    Args:
        atoms (List[MultiLevelAtom]): A list of atomic models.
        # temperature (float): A representative atmospheric temperature [K].
        v_turb (float): Microturbulent velocity [cm/s].
        # max_resol_nm (float): The minimum allowed wavelength spacing [nm]. Points
        #                       closer than this will be merged.
        # min_resol_nm (float): The maximum allowed wavelength spacing [nm]. The grid
        #                       will be densified if points are further apart.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A sorted, unique array of frequencies in Hz [s^-1].
            - The corresponding trapezoidal integration weights.
    """
    frequency_set = set()

    # Generate the combined frequency set from continua and lines
    for atom in atoms:
        # Dynamically fetch grids representing Continua transitions properly (Explicit and Hydrogenic)
        for cont in atom.continua:
            wl_grid_nm = cont.get_wavelength_grid(atom.levels) 
            for wl_nm in wl_grid_nm:
                frequency_set.add(c_CGS / (wl_nm * 1e-7))

        # Generate and add frequencies for spectral lines in Doppler units
        for line in atom.lines:
            if line.nu0 == 0:
                continue

            # Lightweaver N_lambda point mapping: 
            # It uses exactly (N_lambda + 1) // 2 points per half grid.
            n_lambda = (line.quadrature.N_lambda + 1) // 2
            if n_lambda < 2:
                n_lambda = 2

            if line.quadrature.q_wing <= 2.0 * line.quadrature.q_core:
                beta = 1.0
            else:
                beta = line.quadrature.q_wing / (2.0 * line.quadrature.q_core)

            y = beta + np.sqrt(beta**2 + (beta - 1.0) * n_lambda + 2.0 - 3.0 * beta)
            b = 2.0 * np.log(y) / (n_lambda - 1)
            a = line.quadrature.q_wing / (n_lambda - 2.0 + y**2)
            
            nl = np.arange(n_lambda)
            q_half = a * (nl + (np.exp(b * nl) - 1.0))

            x_grid = np.concatenate((-np.flip(q_half[1:]), q_half))

            # v_thermal_rep = np.sqrt(2.0 * kB_CGS * temperature / (m_u_CGS * atom.mass))
            # v_doppler_rep = np.sqrt(v_thermal_rep**2 + v_turb**2)
            # doppler_width_nu_rep = (line.nu0 / c_CGS) * v_doppler_rep

            # for nu in (line.nu0 + x_grid * doppler_width_nu_rep):
            #     frequency_set.add(nu)

            # Lightweaver defines the characteristic line grid velocity as a constant 3 km/s
            v_micro_char = v_turb # 3.0e5 # cm/s
            doppler_width_lambda = line.lambda0 * (v_micro_char / c_CGS)
            
            # The grid is built symmetric in *wavelength* space, not frequency!
            for lam in (line.lambda0 + x_grid * doppler_width_lambda):
                frequency_set.add(c_CGS / lam)

    # reference 500 nm point
    frequency_set.add(c_CGS / (500.0 * 1e-7))

    nus = np.array(sorted(list(frequency_set)))
    n_nu = nus.size
    
    # # Grid refinement (same as original code)
    # if n_nu > 1:
    #     max_resol_cm = max_resol_nm * 1e-7
    #     min_resol_cm = min_resol_nm * 1e-7

    #     pruned_nus = [nus[0]]
    #     for i in range(1, n_nu):
    #         current_nu = pruned_nus[-1]
    #         max_resol_hz = (current_nu**2 / c_CGS) * max_resol_cm
    #         if (nus[i] - current_nu) >= max_resol_hz:
    #             pruned_nus.append(nus[i])
        
    #     final_nus = [pruned_nus[0]]
    #     for i in range(len(pruned_nus) - 1):
    #         start_nu = pruned_nus[i]
    #         end_nu = pruned_nus[i+1]
    #         min_resol_hz = (start_nu**2 / c_CGS) * min_resol_cm

    #         if (end_nu - start_nu) > min_resol_hz:
    #             lambda_end_nm = (c_CGS / start_nu) * 1e7
    #             lambda_start_nm = (c_CGS / end_nu) * 1e7

    #             num_intervals = int(np.ceil((lambda_end_nm - lambda_start_nm) / min_resol_nm))
    #             if num_intervals > 0:
    #                 new_lambdas = np.linspace(lambda_start_nm, lambda_end_nm, num_intervals + 1)
    #                 new_nus = np.flip(c_CGS / (new_lambdas[1:-1] * 1e-7))
    #                 final_nus.extend(new_nus)

    #         final_nus.append(end_nu)
    #     nus = np.array(final_nus)
    # We do NOT remove points. Lightweaver uses the full set of grid points
    # resulting from the lines and continua.
    
    n_nu = nus.size
    if n_nu < 2:
        return nus, np.zeros(n_nu)

    weights = np.zeros(n_nu)
    weights[0] = 0.5 * (nus[1] - nus[0])
    weights[-1] = 0.5 * (nus[-1] - nus[-2])
    weights[1:-1] = 0.5 * (nus[2:] - nus[:-2])

    return nus, weights
