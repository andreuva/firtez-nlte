from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
from constants import *
from debug_functions import print_eq_system
from chemeq import passive_electron_density
import atomic_data

from scipy.interpolate import RectBivariateSpline
from scipy.special import gamma

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

class BarklemTable:
    def __init__(self, cs_flat, al_flat, shape, neff0):
        self.cross = np.array(cs_flat).reshape(shape)
        self.alpha = np.array(al_flat).reshape(shape)
        self.neff1 = neff0[0] + np.arange(shape[0]) * 0.1
        self.neff2 = neff0[1] + np.arange(shape[1]) * 0.1

class Barklem:
    barklem_sp = BarklemTable(atomic_data.cs_sp_flat, atomic_data.al_sp_flat, (46, 43), (1.0, 1.3))
    barklem_pd = BarklemTable(atomic_data.cs_pd_flat, atomic_data.al_pd_flat, (43, 33), (1.3, 2.3))
    barklem_df = BarklemTable(atomic_data.cs_df_flat, atomic_data.al_df_flat, (33, 28), (2.3, 3.3))

def get_barklem_cross_section(atom, line, vals):
    SOrbit = 0
    POrbit = 1
    DOrbit = 2
    FOrbit = 3

    result = [vals[0], vals[1], 0.0]

    if vals[0] < 20.0:
        lowerNum = atom.levels[line.lower_level_index].L
        upperNum = atom.levels[line.upper_level_index].L
        if lowerNum is None or upperNum is None:
            raise ValueError('L not provided for levels.')

        nums = (lowerNum, upperNum)
        if nums == (SOrbit, POrbit) or nums == (POrbit, SOrbit):
            table = Barklem.barklem_sp
        elif nums == (POrbit, DOrbit) or nums == (DOrbit, POrbit):
            table = Barklem.barklem_pd
        elif nums == (DOrbit, FOrbit) or nums == (FOrbit, DOrbit):
            table = Barklem.barklem_df
        else:
            raise ValueError('Not a valid shell combination.')

        Z = 1
        
        # We need the continuum level for limits
        upper_lvl = atom.levels[line.upper_level_index]
        lower_lvl = atom.levels[line.lower_level_index]
        current_ion = upper_lvl.ionization
        cont_level = next((lvl for lvl in atom.levels if lvl.ionization == current_ion + 1), None)
        E_cont = cont_level.energy if cont_level else atom.levels[-1].energy

        deltaEi = E_cont - lower_lvl.energy
        deltaEj = E_cont - upper_lvl.energy
        E_Rydberg = E_Ryd_erg / (1.0 + m_e_CGS / (atom.mass * m_u_CGS))

        neff1 = Z * np.sqrt(E_Rydberg / deltaEi)
        neff2 = Z * np.sqrt(E_Rydberg / deltaEj)

        if nums[0] > nums[1]:
            neff1, neff2 = neff2, neff1

        if not (table.neff1[0] <= neff1 <= table.neff1[-1]):
            raise ValueError(f'neff1 ({neff1}) outside table [{table.neff1[0]}, {table.neff1[-1]}].')
        if not (table.neff2[0] <= neff2 <= table.neff2[-1]):
            raise ValueError(f'neff2 ({neff2}) outside table [{table.neff2[0]}, {table.neff2[-1]}].')

        result[0] = float(RectBivariateSpline(table.neff1, table.neff2, table.cross)(neff1, neff2)[0, 0])
        result[1] = float(RectBivariateSpline(table.neff1, table.neff2, table.alpha)(neff1, neff2)[0, 0])

    reducedMass = m_u_CGS / (1.0 / 1.008 + 1.0 / atom.mass)
    meanVel = np.sqrt(8.0 * kB_CGS / (np.pi * reducedMass))
    sigma = result[0]
    alpha = result[1]
    crossSection = sigma * a0_CGS**2 * (meanVel / 1e6)**(-alpha)

    result[0] = 2.0 * ((4.0 / np.pi)**(alpha / 2.0)
                 * gamma(2.0 - alpha / 2.0) * meanVel * crossSection)
    result[2] = 1.0

    return result


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
    # Passive atoms are held at their LTE populations: they still contribute line and
    # bound-free opacity, but their statistical equilibrium is never solved. This is the
    # standard active/passive distinction of RH and Lightweaver, and it is what the SNAPI
    # benchmark run used for hydrogen (its H populations are bit-identical across all 97
    # iterations while Ca II changes by 264x).
    is_active: bool = True
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
            is_active=bool(data.get('is_active', True)),
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

        # Thermal velocity component: sqrt(2kT/m). v_turb may be a scalar or a per-depth
        # array -- FAL-C's microturbulence runs from 1.8 km/s at the bottom to 6.8 km/s in
        # the chromosphere, comparable to the thermal width, so a single constant cannot
        # represent it (see audit/BENCHMARK.md).
        v_turb_arr = np.asarray(v_turb, dtype=float)
        if v_turb_arr.ndim == 0:
            v_turb_arr = np.full(Ndepth, float(v_turb))
        elif v_turb_arr.size != Ndepth:
            raise ValueError(f"turbulent_velocity has {v_turb_arr.size} entries, expected {Ndepth}")
        v_thermal = np.sqrt(2.0 * kB_CGS * temp / (m_u_CGS * self.mass))
        v_doppler = np.sqrt(v_thermal**2 + v_turb_arr**2)

        for il, line in enumerate(self.lines):
            if line.nu0 > 0:
                self.doppler_widths[:, il] = (line.nu0 / c_CGS) * v_doppler

def validate_abundances(atoms: List[MultiLevelAtom]) -> None:
    """
    Warn where a model atom's abundance disagrees with the chemeq ABUND table.

    Two abundance sets are in play: each model atom's own, which sets N_total in the
    statistical equilibrium, and chemeq's ABUND, which drives the background equation of
    state and the background opacity species. With electron_density_mode="delta" the
    disagreement cancels, because only the atom's DEPARTURE from its own LTE charge is
    applied on top of n_e,bg. With "nlte" it does not: the model atom replaces the EOS's
    contribution for that element outright, so an abundance mismatch shifts n_e directly.
    """
    from chemeq import ABUND

    for atom in atoms:
        eos = 10.0 ** (ABUND[atom.Z - 1] - 12.0)
        if eos <= 0.0:
            continue
        ratio = atom.abundance / eos
        if abs(ratio - 1.0) > 0.02:
            print(f"WARNING: {atom.name}: abundance {atom.abundance:.4e} differs from the "
                  f"chemeq EOS table value {eos:.4e} by {100*(ratio-1):+.1f}%. The two are "
                  f"used for different things (SE particle conservation vs the background "
                  f"EOS and opacities); with electron_density_mode='nlte' the difference "
                  f"biases n_e.")


def validate_ionization_stages(atoms: List[MultiLevelAtom]) -> None:
    """
    Check that each atom's `ionization` labels really are the ionic charge (0 = neutral).

    The code uses `ionization` for three physically distinct things -- the partition-function
    lookup in compute_lte_populations, the charge sum in solve_SEE, and the effective charge
    Z = ionization + 1 in the Stark and van der Waals recipes -- so a model that labels its
    stages 1-based silently produces the wrong partition function, the wrong electron
    contribution and the wrong broadening. That is exactly the class of error F-001 was.

    The labelling is checked against physics, not convention: the energy separating the
    model's highest stage from its lowest must equal the ionization potential of the lowest
    stage, which chemeq tabulates. Hydrogen is special-cased because chemeq's Z=1 row is the
    (H-, H I, H II) triplet, so a neutral H model matches XII rather than XI.

    Warns rather than raises: a mislabelled model still runs, just wrongly, and stopping a
    long synthesis outright is worse than telling the user loudly.
    """
    from chemeq import XI, XII

    for atom in atoms:
        stages = sorted({lvl.ionization for lvl in atom.levels})
        if len(stages) < 2:
            continue
        lo, hi = stages[0], stages[-1]
        e_ground = min(l.energy for l in atom.levels if l.ionization == lo)
        e_head = min(l.energy for l in atom.levels if l.ionization == hi)
        chi_eV = (e_head - e_ground) / eV_CGS

        xi, xii = XI[atom.Z - 1], XII[atom.Z - 1]
        if atom.Z == 1:
            expected = 0 if abs(chi_eV - xii) / xii < 0.02 else None
        elif abs(chi_eV - xi) / xi < 0.02:
            expected = 0
        elif abs(chi_eV - xii) / xii < 0.02:
            expected = 1
        else:
            expected = None

        if expected is None:
            print(f"WARNING: {atom.name}: ionization energy {chi_eV:.4f} eV matches neither "
                  f"XI ({xi:.3f}) nor XII ({xii:.3f}) for Z={atom.Z}; cannot verify the "
                  f"ionization stage labelling.")
        elif expected != lo:
            print(f"WARNING: {atom.name}: levels are labelled ionization={lo} but the "
                  f"ionization energy {chi_eV:.4f} eV identifies the modelled stage as "
                  f"ionization={expected}. Partition functions, the electron charge sum and "
                  f"the broadening effective charge will all be wrong. Relabel the levels.")


VMICRO_CHAR = 3.0e5   # cm/s. RH's VMICRO_CHAR: the fixed characteristic velocity used to
                      # lay out per-line wavelength grids (getlambda.c). It is NOT the
                      # atmosphere's microturbulence -- the real profile width still comes
                      # from atom.doppler_widths. See audit F-018.


def create_frequency_grid(atoms: List[MultiLevelAtom], 
                        #   temperature: float,
                        #   v_turb: float = 0.0,
                        #   max_resol_nm: float = 3e-4,
                        #   min_resol_nm: float = 1e1,) -> Tuple[np.ndarray, np.ndarray]:
                          v_turb: float = 0.0,
                          v_micro_char: float = VMICRO_CHAR) -> Tuple[np.ndarray, np.ndarray]:
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

            # RH and Lightweaver lay the per-line grid out on a FIXED characteristic
            # velocity (RH's VMICRO_CHAR = 3 km/s), not on the atmosphere's microturbulence.
            # An earlier revision of this audit used max(v_turb) here to accommodate a
            # depth-dependent v_turb; for FAL-C that is 6.75 km/s, which stretches every
            # window by 2.25x and halves the number of points sampling the line core
            # (13 vs 25 within +-1 real Doppler width for Ly-alpha). Restored to the RH
            # convention; override via the v_micro_char argument if needed. See audit F-018.
            doppler_width_lambda = line.lambda0 * (v_micro_char / c_CGS)
            
            # The grid is built symmetric in *wavelength* space, not frequency!
            local_lambdas = line.lambda0 + x_grid * doppler_width_lambda
            local_nus = c_CGS / local_lambdas
            for nu in local_nus:
                frequency_set.add(nu)
            line.max_delta_nu = np.max(np.abs(local_nus - line.nu0))

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


def compute_line_frequency_weights(atoms: List[MultiLevelAtom],
                                   frequency_grid: np.ndarray) -> None:
    """
    Build per-transition trapezoidal quadrature weights, restricted to each line's own
    frequency window, and attach them to every Line as `grid_mask` and `freq_weights`.

    Why this is needed
    ------------------
    The weights returned by `create_frequency_grid` are a trapezoid rule over the *union*
    grid: each point's weight straddles half the gap to its global neighbours. Re-using
    them for a *per-line* integral gives the outermost point of a line's window a weight
    that spans the empty gap separating this line's frequency block from the next block in
    the spectrum. That weight can exceed the local grid spacing by three orders of
    magnitude and then dominates int(phi dnu).

    Measured on the unpatched code for H Br-alpha (4052 nm) at T = 1e5 K: the single
    window-edge point carried a weight of 2.88e13 Hz against a local spacing of 4.0e9 Hz
    and contributed 12.22 of the total integral of 13.31 (92%). The subsequent numerical
    renormalization then deflated the line opacity and J-bar by that factor, and turned
    J-bar into a weighted average of I dominated by the window edge -- i.e. the local
    continuum intensity rather than the line radiation field.

    RH (`getlambda.c`) and Lightweaver (`Transition.compute_phi`) both integrate each
    transition over its own subgrid; this reproduces that behaviour on the shared grid.

    See audit/FINDINGS.md F-002.
    """
    for atom in atoms:
        for line in atom.lines:
            mask = np.abs(frequency_grid - line.nu0) <= line.max_delta_nu
            w = np.zeros_like(frequency_grid)
            nu = frequency_grid[mask]
            if nu.size >= 3:
                wl = np.empty(nu.size)
                wl[0] = 0.5 * (nu[1] - nu[0])
                wl[-1] = 0.5 * (nu[-1] - nu[-2])
                wl[1:-1] = 0.5 * (nu[2:] - nu[:-2])
                w[mask] = wl
            elif nu.size == 2:
                w[mask] = 0.5 * (nu[1] - nu[0])
            elif nu.size == 1:
                # Degenerate window: nothing to integrate over. Leave a unit weight so the
                # profile normalization stays finite; such a line is unusable anyway.
                w[mask] = 1.0
            line.grid_mask = mask
            line.freq_weights = w



def init_line_broadening(atoms: List[MultiLevelAtom], atmosphere: "Atmosphere") -> None:
    """
    Precompute the depth-independent broadening constants for every line and attach them to
    the Line objects (`vdw_cross`, `stark_vrel_factor`, `stark_c23`, `lin_stark_factor`,
    `barklem_c0/c1`, `mult_stark_coeff`).

    These were previously set as a side effect of running main.py, which meant
    `formal_solver.get_RT_coefficients` could not be called at all without executing the
    whole driver script -- the direct reason the RT module had no unit tests (audit F-009).
    Extracted verbatim; behaviour is unchanged.
    """
    for atom in atoms:
        atom_mass_CGS = atom.mass * m_u_CGS
        for line in atom.lines:
            # Defaults for depth-dependent terms
            line.vdw_cross = 0.0
            line.stark_vrel_factor = 0.0
            line.stark_c23 = 0.0
            line.lin_stark_factor = 0.0
            line.barklem_c0 = 0.0
            line.barklem_c1 = 0.0
            line.mult_stark_coeff = 0.0

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

                elif elastic.get("type") == "VdwBarklem":
                    '''
                    Implementation of the Barklem method for van der Waals broadening.
                    '''
                    vals = elastic.get("vals", [0.0, 0.0])
                    barklemVals = get_barklem_cross_section(atom, line, vals)
                    line.barklem_c0 = barklemVals[0]
                    line.barklem_c1 = barklemVals[1]

                    # Helium contribution (same Unsold cross section as in Lightweaver)
                    deltaR = (E_Ryd_erg / (E_cont - upper_lvl.energy))**2 - (E_Ryd_erg / (E_cont - lower_lvl.energy))**2
                    Z = 1 # stage + 1 = 0 + 1 = 1 for neutral atom in VdwBarklem
                    C6_CGS = 2.5 * q_e_CGS**2 * alpha_H_CGS * 2.0 * np.pi * (Z * a0_CGS)**2 / h_CGS * abs(deltaR)
                    C625 = C6_CGS**0.4
                    vRel35He = (8.0 * kB_CGS / (np.pi * atom_mass_CGS) * (1.0 + atom_mass_CGS / m_He_CGS))**0.3

                    line.vdw_cross = 8.08 * atmosphere.he_abund * vRel35He * C625

                elif elastic.get("type") == "MultiplicativeStarkBroadening":
                    '''
                    Simple expression for multiplicative Stark broadening, ne * coeff
                    '''
                    line.mult_stark_coeff = elastic.get("coeff", 0.0)

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

ELECTRON_MODES = ("eos", "delta", "nlte")


def solve_SEE(atoms: List[MultiLevelAtom], atmosphere: "Atmosphere",
              max_itterations=100, tolerance=1e-5,
              electron_mode: str = "nlte") -> float:
    """
    Solves the coupled Statistical Equilibrium and Charge/Particle
    Conservation equations using a decoupled iterative Lambda method.

    This function iterates at each depth point:
    1. Solves SE for each atom using the current electron density (ne).
    2. Calculates a new 'ne' based on the new populations.
    3. Repeats until 'ne' converges locally.

    Args:
        atoms: A list of MultiLevelAtom objects.
        atmosphere: The Atmosphere object containing T, P, ne, nh...

    Electron density treatment, selected by `electron_mode`:

      "eos"    n_e is held at the background LTE equation-of-state value and never updated.
               This is SNAPI's convention and RH's conserveCharge=False.

      "delta"  n_e = n_e,bg + (charge of the active atoms now - their charge in LTE).
               The historical behaviour. It is charge conservation, but ANCHORED to the
               background: the electrons donated by everything that is not an active atom
               stay frozen at their LTE value at n_e,bg and never respond to n_e moving.

      "nlte"   Full charge conservation. The active atoms donate from their NLTE populations
               and every other element is re-evaluated from Saha AT THE CURRENT n_e, so the
               passive donors track the solution instead of being frozen. n_H is held fixed,
               as RH does with conserveCharge=True; the pressure-balance feedback of n_e on
               n_H is not followed.

    Returns:
        The maximum relative change in the state vector (populations + ne)
        from the *start* of the call.
    """
    if electron_mode not in ELECTRON_MODES:
        raise ValueError(f"electron_mode must be one of {ELECTRON_MODES}, got {electron_mode!r}")
    active_Z = {atom.Z for atom in atoms}

    # Store old state for final convergence check
    old_ne = atmosphere.ne.copy()
    old_pops = {atom.name: atom.populations.copy() for atom in atoms}

    # --- Main Loop over All Depth Points ---
    for k in range(atmosphere.Ndepth):
        Tk = atmosphere.temp[k]
        kT = kB_CGS * Tk
        nh = atmosphere.nh[k]
        ne_current_iter = atmosphere.ne[k]
        
        if Tk <= 2000 or nh <= 0:
            # Skip this depth, old populations/ne will be preserved
            print(f"Skipping depth k={k} (z={atmosphere.zgrid[k]/1e5}km) due to low T or non-positive n_H.")
            continue
            
        if ne_current_iter <= 0:
            print(f"Warning: Initial ne at depth k={k} is non-positive. Setting to a small positive value for iteration.")
            ne_current_iter = 1e-20

        # Compute initial charge from modelled atoms (before solving SE)
        lte_total_charge = 0.0

        # Save the populations from the outer ALI iteration for S_old computation.
        # These must NOT be overwritten during the local ne-iterations.
        old_pops_for_S = {}
        for atom in atoms:
            charges_arr = np.array([l.ionization for l in atom.levels])
            lte_total_charge += np.sum(atom.lte_populations[k, :] * charges_arr)
            old_pops_for_S[atom.name] = atom.populations[k, :].copy()

        # --- Local Iteration for (Populations <-> ne) ---
        # In "eos" mode the electron density is held at the background EOS value and never
        # updated, so one pass suffices -- nothing couples back into n_e.
        n_local = 1 if electron_mode == "eos" else max_itterations
        for local_iter in range(n_local):
            
            ne_for_rates = ne_current_iter.copy()
            new_total_charge = 0.0
            
            # --- Solve all atom populations with fixed ne ---
            for atom in atoms:
                if not atom.is_active:
                    # Held at LTE: contributes opacity and charge, but no SE solve. Its
                    # charge contribution cancels in delta_charge_total because it equals
                    # the LTE value by construction.
                    charges = np.array([l.ionization for l in atom.levels])
                    new_total_charge += np.sum(atom.populations[k, :] * charges)
                    continue
                N_total_k = atom.abundance * nh
                if N_total_k <= 0:
                    print(f"Warning: Atom {atom.name} has non-positive abundance at depth k={k}. Skipping SE solve for this atom.")
                    continue
                    
                # Solve for this atom at this depth, using old_pops for S_old
                populations_new = solve_atom(atom, k, Tk, ne_for_rates, kT, N_total_k, nh,
                                             old_pops_k=old_pops_for_S[atom.name],
                                             atmosphere_ne_bg_k=atmosphere.ne_bg[k])
                
                # Update atom's population *at this depth*
                atom.populations[k, :] = populations_new
                
                # --- Calculate contribution to charge ---
                charges = np.array([l.ionization for l in atom.levels])
                new_total_charge += np.sum(populations_new * charges)

            if electron_mode == "eos":
                ne_current_iter = atmosphere.ne_bg[k]
                break

            # --- Calculate new ne and check convergence ---
            # Update ne self-consistently from the NLTE populations.
            # The delta-charge correction finds the total change in ionization
            # relative to the LTE background state and adds it on top.
            if electron_mode == "nlte":
                # Full charge conservation: active atoms donate from their NLTE populations,
                # every other element from Saha re-evaluated at the CURRENT n_e.
                ne_target = new_total_charge + passive_electron_density(
                    Tk, nh, ne_current_iter, exclude_Z=active_Z)
            else:
                delta_charge_total = new_total_charge - lte_total_charge
                ne_target = atmosphere.ne_bg[k] + delta_charge_total

            # FIX: Prevent catastrophic cancellation from wiping out trace metal electrons.
            # Metals (Fe, Si, Mg) ensure ne never drops below ~1e-6 of the total Hydrogen density.
            min_metal_ne = 1e-6 * nh
            if ne_target < min_metal_ne:
                # If delta_charge wipes out ne, trust the metal floor or a fraction of the background
                ne_target = max(min_metal_ne, atmosphere.ne_bg[k] * 0.01)
            
            # # Target ne is the baseline + the total change in modeled charge
            # ne_target = atmosphere.ne_bg[k] + delta_charge_total
            # if ne_target <= 1e-20: ne_target = 1e-20
            
            # Apply half-step damping to prevent oscillations
            ne_new = 0.5 * (ne_current_iter + ne_target)
            
            rel_change_ne = np.abs(ne_new - ne_current_iter) / max(ne_current_iter, 1e-20)
            ne_current_iter = ne_new
            
            if rel_change_ne < tolerance:
                break # Local convergence reached

        # Update the atmosphere's 'ne' with the converged value
        atmosphere.ne[k] = ne_current_iter
        
        if electron_mode != "eos" and local_iter == max_itterations - 1:
            print(f"Warning: SE local iteration did not converge at depth k={k}")

    # --- Calculate max *global* relative change ---
    # (change from the start of the *entire* solve_SEE call)
    # Get max change in ne
    avg_ne = 0.5 * (old_ne + atmosphere.ne)
    avg_ne[avg_ne < 1e-20] = 1e-20
    rel_change_ne_vec = np.abs(old_ne - atmosphere.ne) / avg_ne
    max_rel_change = np.max(rel_change_ne_vec)
    
    # Get max change in all populations
    for atom in atoms:
        old_p = old_pops[atom.name]
        new_p = atom.populations
        avg_p = 0.5 * (old_p + new_p)
        avg_p[avg_p < 1e-20] = 1e-20
        
        rel_change_p_vec = np.abs(old_p - new_p) / avg_p
        rel_change_p = np.max(rel_change_p_vec)
        
        if rel_change_p > max_rel_change:
            max_rel_change = rel_change_p
            
    return max_rel_change

# --- Internal helper function to solve SE for one atom at one depth ---
def solve_atom(atom: MultiLevelAtom, k: int, Tk: float, ne: float, kT: float,
                N_total_k: float, nh: float, old_pops_k: np.ndarray = None,
                atmosphere_ne_bg_k: float = 0.0) -> np.ndarray:
    """
    Solves A*n = b for a single atom at depth k using fixed ne.
    """
    Nlevel = len(atom.levels)
    energies = np.array([l.energy for l in atom.levels])
    gs = np.array([l.g for l in atom.levels])
    ionizations = np.array([l.ionization for l in atom.levels])

    A_matrix = np.zeros((Nlevel, Nlevel))
    B_vector = np.zeros(Nlevel) 
    R_matrix = np.zeros((Nlevel, Nlevel))
    C_matrix = np.zeros((Nlevel, Nlevel))

    # --- Collisional Rates ---
    for coll in atom.collisions:
        # Ensure i is the lower level, j is the upper level
        i, j = coll.lower_level_index, coll.upper_level_index
        if energies[i] > energies[j]:
            print(f"Warning, incorrent collisional level order in config file. Atom {atom.name}"\
                    f" C_{i}{j} changed to C_{j}{i}.")
            i, j = j, i
        
        dE = energies[j] - energies[i]
        exp_factor = np.exp(-dE / kT) 
        
        C_rate_coeff = np.interp(Tk, coll.temperatures, coll.rates)
        if C_rate_coeff < 0.0:
            print(f"Warning, negative interpolated collision rate for atom {atom.name}"+\
                    f" C_{i}{j} set to 0.0.")
            C_rate_coeff = 0.0
        
        Cij, Cji = 0.0, 0.0
        
        if coll.type == "Omega":
            # Dimensionless effective collision strength
            C0 = 8.629130462809868e-06 # (ERydberg_CGS / np.sqrt(m_e_CGS) * np.pi * a0_cgs**2 * np.sqrt(8.0 / (np.pi * kB_CGS)))
            Cji = C0 * ne * C_rate_coeff / (gs[j] * np.sqrt(Tk))
            Cij = Cji * (gs[j] / gs[i]) * exp_factor
            
        elif coll.type == "CE":
            # Collisional excitation by electrons
            Cji = C_rate_coeff * ne * (gs[i] / gs[j]) * np.sqrt(Tk) * 1e6
            Cij = Cji * (gs[j] / gs[i]) * exp_factor
            
        elif coll.type == "CI":
            # Collisional ionization by electrons
            Cij = C_rate_coeff * ne * np.exp(-dE / kT) * np.sqrt(Tk) * 1e6
            lte_ratio = atom.lte_populations[k, i] / np.maximum(atom.lte_populations[k, j], 1e-100)
            Cji = Cij * lte_ratio
            
        elif coll.type == "CH":
            # Collisions with neutral hydrogen
            Cij = C_rate_coeff * nh
            if ionizations[i] == ionizations[j]:
                Cji = Cij * (gs[i] / gs[j]) * np.exp(dE / kT)
            else:
                lte_ratio = atom.lte_populations[k, i] / np.maximum(atom.lte_populations[k, j], 1e-100)
                Cji = Cij * lte_ratio
                
        elif coll.type == "CP":
            # Collisions with protons
            Cji = C_rate_coeff * nh
            Cij = Cji * (gs[j] / gs[i]) * exp_factor
        else:
            print(f"WARNING: Unknown collision type '{coll.type}' in atom {atom.name}. Skipping this collision.")
            continue
            
        C_matrix[i, j] += Cij
        C_matrix[j, i] += Cji

    # --- Radiative Rates (Bound-Bound) ---
    for il, line in enumerate(atom.lines):
        i, j = line.lower_level_index, line.upper_level_index
        J_bar = atom.Js[k, il]

        # R_matrix[i, j] += line.Blu * J_bar
        # R_matrix[j, i] += line.Aul + line.Bul * J_bar
        
        # Fetch the new approximate operator, capping it to prevent numerical
        # quadrature overshoot from causing negative transition rates.
        L_star = np.clip(atom.Lambda_star_bar[k, il], 0.0, 0.9999999)

        # Calculate the old Source Function using populations from the
        # OUTER iteration (not the locally-updated ones)
        n_l_old = old_pops_k[i]
        n_u_old = old_pops_k[j]
        
        denom = n_l_old * line.Blu - n_u_old * line.Bul
        if denom > 1e-100:
            S_old = (n_u_old * line.Aul) / denom
        else:
            S_old = 0.0

        # Calculate J_effective, ensuring it remains non-negative to preserve
        # the M-matrix structure of the rate matrix.
        J_eff = max(J_bar - L_star * S_old, 0.0)

        # The preconditioned Rybicki-Hummer rates
        R_matrix[i, j] += line.Blu * J_eff
        R_matrix[j, i] += line.Aul * (1.0 - L_star) + line.Bul * J_eff

    # --- (Bound-Free) ---
    # The recombination rate carries the LTE ratio (n_i/n_k)*, which is exactly linear in
    # n_e (HM2014 eq. 9.10). It was tabulated once, outside the Lambda loop, at ne_bg --
    # but n_e is updated every iteration (and inside this routine's own local iteration),
    # while the collisional rates in the same matrix use the current n_e. Rescale so the
    # whole matrix is evaluated at one consistent electron density (audit F-006).
    ne_scale = ne / atmosphere_ne_bg_k if atmosphere_ne_bg_k > 0.0 else 1.0
    for i_cont, cont in enumerate(atom.continua):
        i, j = cont.lower_level_index, cont.upper_level_index
        R_matrix[i, j] += atom.photoionization_rates[k, i_cont]
        R_matrix[j, i] += atom.recombination_rates[k, i_cont] * ne_scale

    total_departure_rate_from_i = np.sum(R_matrix + C_matrix, axis=1)

    for i in range(Nlevel):
        for j in range(Nlevel):
            A_matrix[i, j] = R_matrix[j, i] + C_matrix[j, i]
        A_matrix[i, i] -= total_departure_rate_from_i[i]
    
    # Dynamically find the index of the most populated level.
    max_pop_idx = np.argmax(old_pops_k)

    # Overwrite the equation for the most populated level with the conservation equation.
    # Scale the row by the typical departure rate of the most populated level to balance the matrix conditioning.
    scale = total_departure_rate_from_i[max_pop_idx]
    if scale <= 1e-100:
        scale = 1.0
    A_matrix[max_pop_idx, :] = scale
    B_vector[max_pop_idx] = N_total_k * scale

    # A_element_names = [f"{atom.name}_level_{i}" for i in range(Nlevel)]
    # print(f"\n DEBUG: Rate matrix A for atom {atom.name} at depth k={k}")
    # print_eq_system(A_matrix, B_vector, A_element_names)

    # --- Solve the linear system A*n = b ---
    try:
        populations_new = np.linalg.solve(A_matrix, B_vector)
        # Check for NaNs or Infs
        if not np.all(np.isfinite(populations_new)):
            raise ValueError("Linear solver returned non-finite values (NaN/Inf).")
        populations_new[populations_new < 0] = 1e-100 # Clamp negatives
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error solving atom SE at depth k={k}: {e}")
        if old_pops_k is not None:
            populations_new = old_pops_k.copy()
        else:
            populations_new = atom.populations[k, :].copy()
    
    # Save the rate matrices for debugging
    if not hasattr(atom, 'R_matrix_all') or atom.R_matrix_all.shape != (atom.populations.shape[0], Nlevel, Nlevel):
        atom.R_matrix_all = np.zeros((atom.populations.shape[0], Nlevel, Nlevel))
        atom.C_matrix_all = np.zeros((atom.populations.shape[0], Nlevel, Nlevel))
    atom.R_matrix_all[k, :, :] = R_matrix
    atom.C_matrix_all[k, :, :] = C_matrix

    return populations_new
