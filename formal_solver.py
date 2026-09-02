import numpy as np
from constants import *
from typing import TYPE_CHECKING, List, Dict, Tuple

from atoms import MultiLevelAtom
from atmosphere import Atmosphere
from chemeq import XII

# ===========================================================================
# Wittmann/Mihalas continuous-opacity tables and vectorized helpers
# ===========================================================================

# Free-free Gaunt factor table  (Wittmann A0, shape 12×11)
_Z4LOG = np.array([0., 1.20412, 1.90849, 2.40824, 2.79588, 3.11261])
_A0 = np.array([
    5.53,5.49,5.46,5.43,5.40,5.25,5.00,4.69,4.48,4.16,3.85,
    4.91,4.87,4.84,4.80,4.77,4.63,4.40,4.13,3.87,3.52,3.27,
    4.29,4.25,4.22,4.18,4.15,4.02,3.80,3.57,3.27,2.98,2.70,
    3.64,3.61,3.59,3.56,3.54,3.41,3.22,2.97,2.70,2.45,2.20,
    3.00,2.98,2.97,2.95,2.94,2.81,2.65,2.44,2.21,2.01,1.81,
    2.41,2.41,2.41,2.41,2.41,2.32,2.19,2.02,1.84,1.67,1.50,
    1.87,1.89,1.91,1.93,1.95,1.90,1.80,1.68,1.52,1.41,1.30,
    1.33,1.39,1.44,1.49,1.55,1.56,1.51,1.42,1.33,1.25,1.17,
    0.90,0.95,1.00,1.08,1.17,1.30,1.32,1.30,1.20,1.15,1.11,
    0.55,0.58,0.62,0.70,0.85,1.01,1.15,1.18,1.15,1.11,1.08,
    0.33,0.36,0.39,0.46,0.59,0.76,0.97,1.09,1.13,1.10,1.08,
    0.19,0.21,0.24,0.28,0.38,0.53,0.76,0.96,1.08,1.09,1.09,
], dtype=np.float64).reshape(12, 11)

# Coulomb correction coefficients for COULX (levels N=0..5)
_COULX_A1 = np.array([0.9916, 1.105, 1.101, 1.101, 1.102, 1.0986])
_COULX_B1 = np.array([2.719e3, -2.375e4, -9.863e3, -5.765e3, -3.909e3, -2.704e3])
_COULX_C1 = np.array([-2.268e10, 4.077e8, 1.035e8, 4.593e7, 2.371e7, 1.229e7])

# He I bound-free: statistical weights, threshold frequencies, energy levels
_HE1_G0   = np.array([1.,3.,1.,9.,3.,3.,1.,9.,20.,3.])
_HE1_FREQ0= np.array([5.9452090e15,1.1528440e15,0.9803331e15,0.8761076e15,
                       0.8147100e15,0.4519048e15,0.4030971e15,0.8321191e15,
                       0.3660215e15,0.3627891e15])
_HE1_CHI0 = np.array([0.,19.819,20.615,20.964,21.217,22.718,22.920,
                       23.006,23.073,23.086])

# Mg I Peach data
_PEACH0 = np.array([
    -42.474,-42.350,-42.109,-41.795,-41.467,-41.159,-40.883,
    -41.808,-41.735,-41.582,-41.363,-41.115,-40.866,-40.631,
    -41.273,-41.223,-41.114,-40.951,-40.755,-40.549,-40.347,
    -45.583,-44.008,-42.957,-42.205,-41.639,-41.198,-40.841,
    -44.324,-42.747,-41.694,-40.939,-40.370,-39.925,-39.566,
    -50.969,-48.388,-46.630,-45.344,-44.355,-43.568,-42.924,
    -50.633,-48.026,-46.220,-44.859,-43.803,-42.957,-42.264,
    -53.028,-49.643,-47.367,-45.729,-44.491,-43.520,-42.736,
    -51.785,-48.352,-46.050,-44.393,-43.140,-42.157,-41.363,
    -52.285,-48.797,-46.453,-44.765,-43.486,-42.480,-41.668,
    -52.028,-48.540,-46.196,-44.507,-43.227,-42.222,-41.408,
    -52.384,-48.876,-46.513,-44.806,-43.509,-42.488,-41.660,
    -52.363,-48.856,-46.493,-44.786,-43.489,-42.467,-41.639,
    -54.704,-50.772,-48.107,-46.176,-44.707,-43.549,-42.611,
    -54.359,-50.349,-47.643,-45.685,-44.198,-43.027,-42.418],
    dtype=np.float64).reshape(15, 7)
_FREQMG   = np.array([1.9341452e15,1.8488510e15,1.1925797e15,7.9804046e14,
                       4.5772110e14,4.1440977e14,4.1113514e14])
_FLOG0_MG = np.array([35.32123,35.19844,35.15334,34.71490,34.31318,
                       33.75728,33.65788,33.64994,33.43947])
_TLG0_MG  = np.array([8.29405,8.51719,8.69951,8.85367,8.98720,9.10498,9.21034])

# Si I Peach data
_PEACH1 = np.array([
    38.136,38.138,38.140,38.141,38.143,38.144,38.144,38.145,38.145,
    37.834,37.839,37.843,37.847,37.850,37.853,37.855,37.857,37.858,
    37.898,37.898,37.897,37.897,37.897,37.896,37.895,37.895,37.894,
    40.737,40.319,40.047,39.855,39.714,39.604,39.517,39.445,39.385,
    40.581,40.164,39.893,39.702,39.561,39.452,39.366,39.295,39.235,
    45.521,44.456,43.753,43.254,42.878,42.580,42.332,42.119,41.930,
    45.520,44.455,43.752,43.251,42.871,42.569,42.315,42.094,41.896,
    55.068,51.783,49.553,47.942,46.723,45.768,44.997,44.360,43.823,
    53.868,50.369,48.031,46.355,45.092,44.104,43.308,42.652,42.100,
    54.133,50.597,48.233,46.539,45.261,44.262,43.456,42.790,42.230,
    54.051,50.514,48.150,46.454,45.176,44.175,43.368,42.702,42.141,
    54.442,50.854,48.455,46.733,45.433,44.415,43.592,42.912,42.340,
    54.320,50.722,48.313,46.583,45.277,44.251,43.423,42.738,42.160,
    55.691,51.965,49.444,47.615,46.221,45.119,44.223,43.478,42.848,
    55.661,51.933,49.412,47.582,46.188,45.085,44.189,43.445,42.813,
    55.973,52.193,49.630,47.769,46.349,45.226,44.314,43.555,42.913,
    55.922,52.141,49.577,47.715,46.295,45.172,44.259,43.500,42.858,
    56.828,52.821,50.110,48.146,46.654,45.477,44.522,43.730,43.061,
    56.657,52.653,49.944,47.983,46.491,45.315,44.360,43.569,42.901],
    dtype=np.float64).reshape(19, 9)
_FREQSI1  = np.array([2.1413750e15,1.9723165e15,1.7879689e15,1.5152920e15,
                       0.5572393e15,5.3295914e14,4.7886458e14,4.7216422e14,4.6185133e14])
_FLOG1_SI = np.array([35.45438,35.30022,35.21799,35.11986,34.95438,
                       33.95402,33.90947,33.80244,33.78835,33.76626,33.70518])
_TLG1_SI  = np.array([8.29405,8.51719,8.69951,8.85367,8.98720,
                       9.10498,9.21034,9.30565,9.39266])

# Si II Peach data
_PEACH2 = np.array([
    -43.8941,-43.8941,-43.8941,-43.8941,-43.8941,-43.8941,
    -42.2444,-42.2444,-42.2444,-42.2444,-42.2444,-42.2444,
    -40.6054,-40.6054,-40.6054,-40.6054,-40.6054,-40.6054,
    -54.2389,-52.2906,-50.8799,-49.8033,-48.9485,-48.2490,
    -50.4108,-48.4892,-47.1090,-46.0672,-45.2510,-44.5933,
    -52.0936,-50.0741,-48.5999,-47.4676,-46.5649,-45.8246,
    -51.9548,-49.9371,-48.4647,-47.3340,-46.4333,-45.6947,
    -54.2407,-51.7319,-49.9178,-48.5395,-47.4529,-46.5709,
    -52.7355,-50.2218,-48.4059,-47.0267,-45.9402,-45.0592,
    -53.5387,-50.9189,-49.0200,-47.5750,-46.4341,-45.5082,
    -53.2417,-50.6234,-48.7252,-47.2810,-46.1410,-45.2153,
    -53.5097,-50.8535,-48.9263,-47.4586,-46.2994,-45.3581,
    -54.0561,-51.2365,-49.1980,-47.6497,-46.4302,-45.4414,
    -53.8469,-51.0256,-48.9860,-47.4368,-46.2162,-45.2266],
    dtype=np.float64).reshape(14, 6)
_FREQSI2  = np.array([4.9965417e15,3.9466738e15,1.5736321e15,1.5171539e15,
                       9.2378947e14,8.3825004e14,7.6869872e14])
_FLOG2_SI = np.array([36.32984,36.14752,35.91165,34.99216,34.95561,
                       34.45941,34.36234,34.27572,34.20161])
_TLG2_SI  = np.array([9.21034,9.39266,9.54681,9.68034,9.79813,9.90349])

# Fe I data
_FE1_G1   = np.array([25.,35.,21.,15.,9.,35.,33.,21.,27.,49.,9.,21.,
                       27.,9.,9.,25.,33.,15.,35.,3.,5.,11.,15.,13.,
                       15.,9.,21.,15.,21.,25.,35.,9.,5.,45.,27.,21.,
                       15.,21.,15.,25.,21.,35.,5.,15.,45.,35.,55.,25.])
_FE1_E1   = np.array([500.,7500.,12500.,17500.,19000.,19500.,19500.,
                       21000.,22000.,23000.,23000.,24000.,24000.,24500.,
                       24500.,26000.,26500.,26500.,27000.,27500.,28500.,
                       29000.,29500.,29500.,29500.,30000.,31500.,31500.,
                       33500.,33500.,34000.,34500.,34500.,35000.,35500.,
                       37000.,37000.,37000.,38500.,40000.,40000.,41000.,
                       41000.,43000.,43000.,43000.,43000.,44000.])
_FE1_WNO1 = np.array([63500.,58500.,53500.,59500.,45000.,44500.,44500.,
                       43000.,58000.,41000.,54000.,40000.,40000.,57500.,
                       55500.,38000.,57500.,57500.,37000.,54500.,53500.,
                       55000.,34500.,34500.,34500.,34000.,32500.,32500.,
                       32500.,32500.,32000.,29500.,29500.,31000.,30500.,
                       29000.,27000.,54000.,27500.,24000.,47000.,23000.,
                       44000.,42000.,42000.,21000.,42000.,42000.])


def _coulff_vec(TLOG: float, FREQLG: np.ndarray, NZ: int) -> np.ndarray:
    """Vectorized free-free Gaunt factor. NZ scalar, FREQLG array -> array."""
    GAMLOG = 10.39638 - TLOG / 1.15129 + _Z4LOG[NZ - 1]
    IGAM   = int(np.clip(np.floor(GAMLOG + 7.0), 1, 10))
    P      = GAMLOG - (IGAM - 7)

    HVKTLG = (FREQLG - TLOG) / 1.15129 - 20.63764
    IHVKT  = np.clip(np.floor(HVKTLG + 9.0).astype(int), 1, 11)
    Q      = HVKTLG - (IHVKT - 9)

    i0, i1 = IHVKT - 1, IHVKT          # row indices (arrays)
    j0, j1 = IGAM - 1, IGAM            # col indices (scalars)
    return ((1.0 - P) * ((1.0 - Q) * _A0[i0, j0] + Q * _A0[i1, j0])
            +      P  * ((1.0 - Q) * _A0[i0, j1] + Q * _A0[i1, j1]))


def _coulx_vec(freq: np.ndarray, Z: float = 1.0, nlevels: int = 8) -> np.ndarray:
    """Hydrogenic bound-free cross-sections for levels N=0..nlevels-1. Returns (nlevels, Nfreq)."""
    FREQ1 = freq * 1.0e-10
    out   = np.zeros((nlevels, len(freq)))
    for N in range(nlevels):
        n2        = (N + 1.0) ** 2
        threshold = Z * Z * 3.28805e15 / n2
        above     = freq >= threshold
        if not np.any(above):
            continue
        f1  = FREQ1[above]
        clx = 0.2815 / (f1 ** 3 * n2 * n2 * (N + 1.0)) * Z ** 4
        if N < 6:
            zz_f1 = Z * Z / f1
            clx  *= _COULX_A1[N] + (_COULX_B1[N] + _COULX_C1[N] * zz_f1) * zz_f1
        out[N, above] = clx
    return out


def _mg1op(FREQ, FREQLG, T, TLOG):
    """Mg I Peach cross-section. Port of wittmann.Mg1OP."""
    if FREQ <= _FREQMG[-1]:   # below minimum table frequency
        return 0.0
    NT = int(np.clip(np.floor(T / 1000.0) - 3, 1, 6))
    DT = (TLOG - _TLG0_MG[NT - 1]) / (_TLG0_MG[NT] - _TLG0_MG[NT - 1])
    N = 0
    for n_idx in range(7):
        if FREQ > _FREQMG[n_idx]:
            N = n_idx
            break
    D = (FREQLG - _FLOG0_MG[N]) / (_FLOG0_MG[N + 1] - _FLOG0_MG[N])
    if N > 1:
        N = 2 * N - 1
    D1 = 1.0 - D
    XWL1 = _PEACH0[N + 1, NT - 1] * D + _PEACH0[N, NT - 1] * D1
    XWL2 = _PEACH0[N + 1, NT    ] * D + _PEACH0[N, NT    ] * D1
    return np.exp(XWL1 * (1.0 - DT) + XWL2 * DT)


def _si1op(FREQ, FREQLG, T, TLOG):
    """Si I Peach cross-section. Port of wittmann.Si1OP."""
    if FREQ <= _FREQSI1[-1]:  # below minimum table frequency
        return 0.0
    NT = int(np.clip(np.floor(T / 1000.0) - 3, 1, 8))
    DT = (TLOG - _TLG1_SI[NT - 1]) / (_TLG1_SI[NT] - _TLG1_SI[NT - 1])
    N = 0
    for n_idx in range(9):
        if FREQ > _FREQSI1[n_idx]:
            N = n_idx
            break
    D = (FREQLG - _FLOG1_SI[N]) / (_FLOG1_SI[N + 1] - _FLOG1_SI[N])
    if N > 1:
        N = 2 * N - 1
    D1 = 1.0 - D
    XWL1 = _PEACH1[N + 1, NT - 1] * D + _PEACH1[N, NT - 1] * D1
    XWL2 = _PEACH1[N + 1, NT    ] * D + _PEACH1[N, NT    ] * D1
    return np.exp(-(XWL1 * (1.0 - DT) + XWL2 * DT)) * 9.0


def _si2op(FREQ, FREQLG, T, TLOG):
    """Si II Peach cross-section. Port of wittmann.Si2OP."""
    if FREQ <= _FREQSI2[-1]:  # below minimum table frequency
        return 0.0
    NT = int(np.clip(np.floor(T / 2000.0) - 4, 1, 5))
    DT = (TLOG - _TLG2_SI[NT - 1]) / (_TLG2_SI[NT] - _TLG2_SI[NT - 1])
    N = 0
    for n_idx in range(7):
        if FREQ > _FREQSI2[n_idx]:
            N = n_idx
            break
    D = (FREQLG - _FLOG2_SI[N]) / (_FLOG2_SI[N + 1] - _FLOG2_SI[N])
    if N > 1:
        N = 2 * N - 2
    if N == 13:
        N = 12
    D1 = 1.0 - D
    XWL1 = _PEACH2[N + 1, NT - 1] * D + _PEACH2[N, NT - 1] * D1
    XWL2 = _PEACH2[N + 1, NT    ] * D + _PEACH2[N, NT    ] * D1
    return np.exp(XWL1 * (1.0 - DT) + XWL2 * DT) * 6.0


def plank(frequency, temp):
    """
    Computes the Planck function B_nu(T) in a numerically stable way.
    """
    Bnu = np.zeros_like(frequency)
    if temp <= 0.0:
        return Bnu
    
    hnu_kt = h_CGS * frequency / (kB_CGS * temp)
    OVERFLOW_LIMIT = 200.0

    # Use the Wien approximation: B_nu = (2*h*nu^3 / c^2) * exp(-h*nu / k*T)
    large_arg = (hnu_kt >= OVERFLOW_LIMIT)
    if np.any(large_arg):
        prefactor = 2.0 * h_CGS * frequency[large_arg]**3 / c_CGS**2
        Bnu[large_arg] = prefactor * np.exp(-hnu_kt[large_arg])

    # and avoids catastrophic cancellation when x is very small.
    normal_arg = ~large_arg
    if np.any(normal_arg):
        prefactor = 2.0 * h_CGS * frequency[normal_arg]**3 / c_CGS**2
        Bnu[normal_arg] = prefactor / np.expm1(hnu_kt[normal_arg])
        
    return Bnu

def voigt(v, a):
    """
    Compute the Voigt profile for a list of frequencies (normalized)
    """

    ss = np.abs(v)+a
    dd = (.195e0*np.abs(v)-.176e0)
    zz = a - 1.j*v
    res = v*0.j

    # Run over frequencies
    for i,s,z,d in zip(range(len(ss)),ss,zz,dd):

        if s >= .15e2:
            t = .5641896e0*z/(.5+z*z)
        else:

            if s >= .55e1:

                u = z*z
                t = z*(.1410474e1 + .5641896e0*u)/(.75e0 + u*(.3e1 + u))

            else:

                if a >= d:
                    nt = .164955e2 + z*(.2020933e2 + z*(.1196482e2 +
                                            z*(.3778987e1 + .5642236e0*z)))
                    dt = .164955e2 + z*(.3882363e2 + z*(.3927121e2 +
                                            z*(.2169274e2 + z*(.6699398e1 + z))))
                    t = nt / dt
                else:
                    u = z*z
                    x = z*(.3618331e5 - u*(.33219905e4 - u*(.1540787e4 -
                               u*(.2190313e3 - u*(.3576683e2 - u*(.1320522e1 -
                                  .56419e0*u))))))
                    y = .320666e5 - u*(.2432284e5 - u*(.9022228e4 -
                                       u*(.2186181e4 - u*(.3642191e3 - u*(.6157037e2 -
                                          u*(.1841439e1 - u))))))
                    t = np.exp(u) - x/y
        res[i] = t

    return res



def line_damping(line, atom, iz: int, atmosphere, nH_ground: float) -> float:
    """
    Total damping rate Gamma [s^-1] for one line at one depth: the sum of the natural width
    and every configured elastic (collisional) broadening term.

    Gamma is a *full* width in s^-1, so the Voigt damping parameter is a = Gamma/(4 pi dnuD)
    (the Lorentzian HWHM in ordinary frequency is Gamma/4pi).
    """
    total = 0.0
    for natural in line.broadening.natural:
        if natural.get("type") == "RadiativeBroadening":
            total += natural.get("gamma", 0.0)
        else:
            raise NotImplementedError(f"Natural broadening type {natural.get('type')} not implemented.")

    T = atmosphere.temp[iz]
    ne = atmosphere.ne[iz]
    for elastic in line.broadening.elastic:
        etype = elastic.get("type")
        if etype == "VdwUnsold":
            total += line.vdw_cross * T**0.3 * nH_ground
        elif etype == "QuadraticStarkBroadening":
            total += line.stark_c23 * line.stark_vrel_factor * T**(1.0/6.0) * ne
        elif etype == "HydrogenLinearStarkBroadening":
            total += line.lin_stark_factor * ne**(2.0/3.0)
        elif etype == "VdwBarklem":
            total += (line.barklem_c0 * T**(0.5 * (1.0 - line.barklem_c1))
                      + line.vdw_cross * T**0.3) * nH_ground
        elif etype == "MultiplicativeStarkBroadening":
            total += line.mult_stark_coeff * ne
        else:
            raise NotImplementedError(f"Elastic broadening type {etype} not implemented.")
    return total


def line_profile(line, atom, iz: int, freq_grid: np.ndarray, atmosphere, nH_ground: float) -> np.ndarray:
    """
    Normalised Voigt profile phi_nu [Hz^-1] on the global frequency grid, zero outside the
    line's own window.

    Single source of truth for the profile: main.py's J-bar/Lambda*-bar loop and
    get_RT_coefficients must use bit-identical profiles or the MALI preconditioning breaks
    (the opacity ratio chi_line/chi_total would be inconsistent with the operator it
    weights). These were previously two verbatim copies (audit F-009).

    Normalisation uses the per-line quadrature weights, not the global ones -- see
    atoms.compute_line_frequency_weights and audit F-002.
    """
    dnuD = atom.doppler_widths[iz, line_index(atom, line)]
    a_damp = line_damping(line, atom, iz, atmosphere, nH_ground) / (4.0 * np.pi * dnuD)
    voigt_line = voigt((freq_grid - line.nu0) / dnuD, a_damp).real
    voigt_line[~line.grid_mask] = 0.0
    return voigt_line / np.sum(voigt_line * line.freq_weights)


def line_index(atom, line) -> int:
    """Index of `line` within `atom.lines` (needed to address the doppler_widths table)."""
    for il, ln in enumerate(atom.lines):
        if ln is line:
            return il
    raise ValueError("line does not belong to this atom")

# --------------------------------------------------------------------------
# RT coefficients and continiuum opacities
def get_RT_coefficients(iz: int, freq_grid: np.ndarray, weigths_freq_grid: np.ndarray,
                        atoms: List[MultiLevelAtom],
                        atmosphere: Atmosphere):
    
    emis = np.zeros_like(freq_grid)
    abs = np.zeros_like(freq_grid)

    h_atom = next((a for a in atoms if a.name == "H"), None)
    # True ground state hydrogen mapping. Falls back to background total H if not existing in config.
    nHGround = h_atom.populations[iz, 0] if h_atom else atmosphere.nh[iz]

    for atom in atoms:

        # Add the line contributions (Bound-Bound)
        for il, line in enumerate(atom.lines):
            
            voigt_norm = line_profile(line, atom, iz, freq_grid, atmosphere, nHGround)

            n_u = atom.populations[iz, line.upper_level_index]
            n_l = atom.populations[iz, line.lower_level_index]

            # TO CHECK: efect of using line.nu0 instead of freq_grid
            emis += (h_CGS*line.nu0/(4*np.pi))* n_u * line.Aul * voigt_norm
            abs +=  (h_CGS*line.nu0/(4*np.pi))* voigt_norm * (n_l*line.Blu - n_u*line.Bul)

        # Add the continuum contributions (Bound-Free)
        for ic, cont in enumerate(atom.continua):
            
            n_l = atom.populations[iz, cont.lower_level_index]
            n_u = atom.populations[iz, cont.upper_level_index]
            n_l_star = atom.lte_populations[iz, cont.lower_level_index]
            n_u_star = atom.lte_populations[iz, cont.upper_level_index]

            # Get cross-section already interpolated on the frequency grid
            sigma_nu = atom.photoionization_alphas[ic, :]

            hnu_over_kT = h_CGS * freq_grid / (kB_CGS * atmosphere.temp[iz])  # dimensionless
            stim_factor = np.exp(-hnu_over_kT)  # exp(-h*nu / (kB*T))  -- correct parenthesization
            emis += (2*h_CGS*freq_grid**3/c_CGS**2) * sigma_nu * (n_l_star/n_u_star) *\
                  n_u * stim_factor
            abs +=  sigma_nu * \
                (n_l - n_u*(n_l_star/n_u_star)*stim_factor)

    # Add continuum contribution
    emis_c, abs_c = add_background_opacity(iz, freq_grid, atoms, atmosphere)
    emis += emis_c
    abs += abs_c
    
    # Sanity check for negative absorptions or emissivities
    if np.any(emis < 0):
        print(f"Warning: Negative emission at depth {iz}.")
        # emis = np.maximum(emis, 0)
    if np.any(abs < 0):
        print(f"Warning: Negative absorption at depth {iz}.")
        # abs = np.maximum(abs, vacuum_CGS)

    return emis, abs


def add_background_opacity(iz: int,
                            freq_grid: np.ndarray,
                            atoms: List[MultiLevelAtom],
                            atmosphere: Atmosphere) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes background continuum (emis_c, abs_c) using the complete
    Wittmann/Mihalas opacity suite.

    Thermal absorption emits as kappa * B_nu; coherent scattering (Thomson + Rayleigh)
    emits as sigma * J_nu, NOT sigma * B_nu, so the continuum source function is
    (kappa*B + sigma*J)/(kappa + sigma). J_nu is the mean intensity from the previous
    Lambda iteration -- see audit F-007 for the convergence implication.
    """
    T   = atmosphere.temp[iz]
    ne  = atmosphere.ne[iz]
    B   = plank(freq_grid, T)
    sp  = atmosphere.bg_species

    TKEV   = kB_CGS * T / eV_CGS
    TLOG   = np.log(T)
    HTK    = h_CGS / (kB_CGS * T)
    EHVKT  = np.exp(-freq_grid * HTK)
    STIM   = 1.0 - EHVKT
    FREQLG = np.log(freq_grid)
    FREQ15 = freq_grid * 1.0e-15

    n_H1  = sp['n_HI_per_U'][iz]
    n_H2  = sp['n_HII'][iz]
    n_Hm  = sp['n_Hminus'][iz]
    n_He1 = sp['n_HeI_per_U'][iz]
    n_He2 = sp['n_HeII_per_U'][iz]
    n_He3 = sp['n_HeIII'][iz]

    # `atoms` is the list of ACTIVE NLTE atoms. Their bound-free transitions are already
    # added by get_RT_coefficients, so they must not also be supplied by the LTE background
    # (audit F-017).
    h_atom = next((a for a in atoms if a.name == "H"), None)
    n_H_bf_active = sum(1 for l in h_atom.levels if l.ionization == 0) if h_atom else 0
    ca_active = any(a.name.startswith("Ca") for a in atoms)

    kappa = np.zeros_like(freq_grid)
    kappa += _opac_h_hydrogenic(freq_grid, FREQLG, T, TLOG, TKEV, HTK, EHVKT, STIM, ne, n_H1, n_H2,
                                skip_bf_levels=n_H_bf_active)
    kappa += _opac_h_minus_wittmann(freq_grid, T, TKEV, ne, EHVKT, n_H1, n_Hm)
    kappa += _opac_h2plus(freq_grid, FREQLG, FREQ15, TKEV, STIM, n_H1, n_H2)
    kappa += _opac_he1(freq_grid, FREQLG, T, TLOG, TKEV, EHVKT, STIM, ne, n_He1, n_He2)
    kappa += _opac_he2(freq_grid, FREQLG, T, TLOG, TKEV, EHVKT, STIM, ne, n_He2, n_He3)
    kappa += _opac_he_minus_ff(freq_grid, T, ne, n_He1)

    if T < 12000.0:
        kappa += _opac_metals_cool(freq_grid, FREQLG, T, TLOG, TKEV, HTK, STIM,
                                    sp['n_CI_per_U'][iz], sp['n_MgI_per_U'][iz],
                                    sp['n_AlI_per_U'][iz], sp['n_SiI_per_U'][iz],
                                    sp['n_FeI_per_U'][iz])
    if T < 30000.0:
        kappa += _opac_metals_luke(freq_grid, FREQLG, T, TLOG, TKEV, STIM,
                                    sp['n_NI_per_U'][iz], sp['n_OI_per_U'][iz],
                                    sp['n_MgII_per_U'][iz], sp['n_SiII_per_U'][iz],
                                    0.0 if ca_active else sp['n_CaII_per_U'][iz])

    sigma  = 0.6653e-24 * ne
    sigma += _opac_rayleigh_h(freq_grid, n_H1)
    sigma += _opac_rayleigh_he(freq_grid, n_He1)
    sigma += _opac_rayleigh_h2(freq_grid, T, TLOG, TKEV, n_H1)

    # total  = kappa + sigma
    # return total * B, total
    opp = kappa + sigma
    # TO CHECK: Difference between J_nu and B?
    ems = kappa*B + sigma*atmosphere.J_nu[iz, :]
    return ems, opp


# ===========================================================================
# Wittmann opacity functions  (vectorized numpy, CGS)
# ===========================================================================

def _opac_h_hydrogenic(freq, FREQLG, T, TLOG, TKEV, HTK, EHVKT, STIM,
                        ne, n_H1, n_H2, skip_bf_levels=0):
    """
    H bound-free (8 levels, Coulomb) + H free-free.  Port of wittmann.HOP.

    `skip_bf_levels` omits the bound-free edges of the n = 1 .. skip_bf_levels levels,
    for use when hydrogen is an ACTIVE NLTE atom whose own continua are already added by
    get_RT_coefficients. Without this the same hydrogen bound-free opacity is counted
    twice -- measured as a factor ~2.0 in the Lyman continuum (audit F-017). The free-free
    term and the higher levels are always retained, since the active model does not carry
    them. RH and Lightweaver exclude active atoms from the background for the same reason.
    """
    FREQ3 = (freq * 1.0e-10) ** 3
    CFREE = 3.6919e-22 / FREQ3
    FREET = ne * CFREE * n_H2 / np.sqrt(T)
    n1    = (np.arange(1, 9, dtype=float)) ** 2
    BOLT  = np.exp(-13.595 * (1.0 - 1.0/n1) / TKEV) * 2.0 * n1 * n_H1
    CONT  = _coulx_vec(freq, Z=1.0)
    XR     = n_H1 / 13.595 * TKEV
    BOLTEX = np.exp(-13.427 / TKEV) * XR
    EXLIM  = np.exp(-13.595 / TKEV) * XR
    BOLTEX = np.where(freq < 4.05933e13, EXLIM / np.maximum(EHVKT, 1.0e-300), BOLTEX)
    C     = 0.2815 / FREQ3
    GAUNT = _coulff_vec(TLOG, FREQLG, NZ=1)
    H  = (CONT[6]*BOLT[6] + CONT[7]*BOLT[7]
          + (BOLTEX - EXLIM)*C + GAUNT*FREET) * STIM
    lo = min(int(skip_bf_levels), 6)          # CONT[:6] holds n = 1..6
    if lo < 6:
        H += np.sum(CONT[lo:6] * BOLT[lo:6, np.newaxis], axis=0) * (1.0 - EHVKT)
    return np.maximum(H, 0.0)


def _opac_h_minus_wittmann(freq, T, TKEV, ne, EHVKT, n_H1, n_Hm):
    """H- bound-free + free-free.  Port of wittmann.HMINOP."""
    FREQ1 = freq * 1.0e-10
    B_ff  = (1.3727e-15 + 4.3748 / freq) / FREQ1
    C_ff  = -2.5993e-7 / FREQ1 ** 2
    HMINFF = (B_ff + C_ff / T) * n_H1 * ne * 2.0e-20
    HMINBF = np.zeros_like(freq)
    hi = freq >= 2.111e14
    lo = (freq > 1.8259e14) & ~hi
    HMINBF[lo] = 3.695e-6 + (-1.251e-1 + 1.052e3/FREQ1[lo]) / FREQ1[lo]
    HMINBF[hi] = (6.801e-10 + (5.358e-3 + (1.481e3 + (-5.519e7
                  + 4.808e11/FREQ1[hi])/FREQ1[hi])/FREQ1[hi])/FREQ1[hi])
    HMIN = n_Hm if T < 7730.0 else (
           np.exp(0.7552/TKEV)/(2.0*2.4148e15*T*np.sqrt(T)) * n_H1 * ne)
    return np.maximum(HMINBF * (1.0 - EHVKT) * HMIN * 1.0e-10 + HMINFF, 0.0)


def _opac_h2plus(freq, FREQLG, FREQ15, TKEV, STIM, n_H1, n_H2):
    """H2+ dissociative absorption.  Port of wittmann.H2PLOP."""
    out  = np.zeros_like(freq)
    mask = freq <= 3.28805e15
    if not np.any(mask):
        return out
    flg  = FREQLG[mask]; f15 = FREQ15[mask]
    FR   = (-3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3*flg)*flg)*flg)*flg)
    ES   = (-7.342e-3 + (-2.409 + (1.028 + (-0.4230 + (0.1224 - 0.01351*f15)*f15)*f15)*f15)*f15)
    out[mask] = np.exp(-ES/TKEV + FR) * 2.0 * n_H1 * n_H2 * STIM[mask]
    return np.maximum(out, 0.0)


def _opac_he1(freq, FREQLG, T, TLOG, TKEV, EHVKT, STIM, ne, n_He1, n_He2):
    """He I bound-free (10 transitions) + free-free.  Port of wittmann.HE1OP."""
    FREQ3 = (freq * 1.0e-10) ** 3
    CFREE = 3.6919e8 / FREQ3
    C     = 2.815e-1 / FREQ3
    FREET = ne * 1.0e-10 * n_He2 * 1.0e-10 / np.sqrt(T) * 1.0e-10
    BOLT  = (np.exp(-_HE1_CHI0[:, np.newaxis] / TKEV)
             * _HE1_G0[:, np.newaxis] * n_He1)
    XRLOG  = np.log(max(n_He1 * (2.0/13.595) * TKEV, 1.0e-300))
    BOLTEX = np.exp(-23.730/TKEV + XRLOG)
    EXLIM  = np.exp(-24.587/TKEV + XRLOG)
    EX     = np.where(freq < 2.055e14, EXLIM/np.maximum(EHVKT, 1.0e-300), BOLTEX)
    dum = np.array([33.32 - 2.0*FREQLG,
                    -390.026 + (21.035 - 0.318*FREQLG)*FREQLG,
                    26.83 - 1.91*FREQLG,  61.21 - 2.9*FREQLG,
                    81.35 - 3.5*FREQLG,  12.69 - 1.54*FREQLG,
                    23.85 - 1.86*FREQLG, 49.30 - 2.60*FREQLG,
                    85.20 - 3.69*FREQLG, 58.81 - 2.89*FREQLG])
    TRANS = np.zeros_like(dum)
    for nmin_idx in range(10):
        if np.any(freq >= _HE1_FREQ0[nmin_idx]):
            above = freq >= _HE1_FREQ0[nmin_idx]
            TRANS[nmin_idx:, above] = np.exp(np.clip(dum[nmin_idx:, above], -300.0, 300.0))
            break
    GAUNT = _coulff_vec(TLOG, FREQLG, NZ=1)
    HE1 = ((EX - EXLIM)*C + np.sum(TRANS*BOLT, axis=0)
           + GAUNT*FREET*CFREE) * STIM
    return np.maximum(HE1, 0.0)


def _opac_he2(freq, FREQLG, T, TLOG, TKEV, EHVKT, STIM, ne, n_He2, n_He3):
    """He II hydrogenic opacity.  Port of wittmann.HE2OP."""
    FREQ3 = (freq * 1.0e-5) ** 3
    CFREE = 3.6919e-7 / FREQ3 * 4.0
    C     = 2.815e14 * 4.0 / FREQ3
    FREET = ne * n_He3 / np.sqrt(T)
    N12   = (np.arange(1, 10, dtype=float)) ** 2
    BOLT  = (np.exp(-(54.403 - 54.403/N12[:, np.newaxis])/TKEV)
             * 2.0 * N12[:, np.newaxis] * n_He2)
    CONT  = _coulx_vec(freq, Z=2.0, nlevels=9)
    XR     = n_He2 / 13.595 * TKEV
    BOLTEX = np.exp(-53.859/TKEV) * XR
    EXLIM  = np.exp(-54.403/TKEV) * XR
    EX     = np.where(freq < 1.31522e14, EXLIM/np.maximum(EHVKT, 1.0e-300), BOLTEX)
    GAUNT  = _coulff_vec(TLOG, FREQLG, NZ=2)
    HE2    = ((EX - EXLIM)*C + np.sum(CONT*BOLT, axis=0)
              + GAUNT*CFREE*FREET) * STIM
    return np.where(HE2 >= 1.0e-20, HE2, 0.0)


def _opac_he_minus_ff(freq, T, ne, n_He1):
    """He- free-free.  Port of wittmann.HEMIOP."""
    A =  3.397e-26 + (-5.216e-11 + 7.039e5 /freq)/freq
    B = -4.116e-22 + ( 1.067e-6  + 8.135e9 /freq)/freq
    C =  5.081e-17 + (-8.724e-3  - 5.659e12/freq)/freq
    return np.maximum((A*T + B + C/T) * ne * n_He1 * 1.0e-20, 0.0)


def _opac_rayleigh_h(freq, n_H1):
    """Rayleigh scattering by H I.  Port of wittmann.HRAYOP."""
    WAVE  = np.minimum(freq, 2.463e15)
    WAVE  = 2.997925e18 / WAVE
    WW    = WAVE * WAVE
    return (5.799e-13 + 1.422e-6/WW + 2.784/WW**2) / WW**2 * n_H1 * 2.0


def _opac_rayleigh_he(freq, n_He1):
    """Rayleigh scattering by He I.  Port of wittmann.HERAOP."""
    WW  = (2.997925e3 / np.minimum(freq * 1.0e-15, 5.15)) ** 2
    arg = 1.0 + (2.44e5 + 5.94e10/(WW - 2.90e5)) / WW
    return 5.484e-14 / WW**2 * arg**2 * n_He1


def _opac_rayleigh_h2(freq, T, TLOG, TKEV, n_H1):
    """Rayleigh scattering by H2.  Port of wittmann.H2RAOP."""
    WW   = (2.997925e18 / np.minimum(freq, 2.922e15)) ** 2
    sig  = (8.14e-13 + 1.28e-6/WW + 1.61/WW**2) / WW**2
    ARG  = (4.477/TKEV - 4.6628e1
            + (1.8031e-3 + (-5.023e-7 + (8.1424e-11 - 5.0501e-15*T)*T)*T)*T
            - 1.5*TLOG)
    return sig * np.exp(ARG) * (n_H1*2.0)**2 if ARG > -80.0 else np.zeros_like(freq)


def _seaton(FREQ0, XSECT, POWER, A, freq):
    """Seaton photoionization cross-section."""
    ratio = FREQ0 / freq
    return XSECT * (A + (1.0-A)*ratio) * ratio ** (int(2.0*POWER + 0.01) // 2)


def _opac_metals_cool(freq, FREQLG, T, TLOG, TKEV, HTK, STIM,
                       n_C1, n_Mg1, n_Al1, n_Si1, n_Fe1):
    """C I, Mg I, Al I, Si I, Fe I.  Port of wittmann.COOLOP. Active T<12000 K."""
    out = np.zeros_like(freq)
    # C I
    C1240 = 5.0 * np.exp(-1.264/TKEV); C1444 = np.exp(-2.683/TKEV)
    C1 = np.zeros_like(freq)
    m = freq >= 2.7254e15; C1[m] += _seaton(2.7254e15, 1.219e-17, 2.0, 3.317, freq[m]) * 9.0
    m = freq >= 2.4196e15; C1[m] += _seaton(2.4196e15, 1.030e-17, 1.5, 2.789, freq[m]) * C1240
    m = freq >= 2.0761e15; C1[m] += _seaton(2.0761e15, 9.590e-18, 1.5, 3.501, freq[m]) * C1444
    out += C1 * n_C1
    # Mg I (Peach table, scalar loop)
    Mg1 = np.array([_mg1op(f, flg, T, TLOG) for f, flg in zip(freq, FREQLG)])
    out += Mg1 * n_Mg1
    # Al I
    out += np.where(freq > 1.443e15, 2.1e-17*(1.443e15/freq)**3 * 6.0 * n_Al1, 0.0)
    # Si I (Peach table)
    Si1 = np.array([_si1op(f, flg, T, TLOG) for f, flg in zip(freq, FREQLG)])
    out += Si1 * n_Si1
    # Fe I
    BOLT_Fe = _FE1_G1 * np.exp(-_FE1_E1 * 2.99792458e10 * HTK)
    for i, wn in enumerate(freq / 2.99792458e10):
        if wn < 21000.0:
            continue
        XXX  = ((_FE1_WNO1 + 3000.0 - wn) / _FE1_WNO1 / 0.1)
        XSEC = np.where(_FE1_WNO1 < wn, 3.0e-18/(1.0 + XXX**4), 0.0)
        out[i] += np.dot(XSEC, BOLT_Fe) * n_Fe1
    return out * STIM


def _opac_metals_luke(freq, FREQLG, T, TLOG, TKEV, STIM,
                       n_N1, n_O1, n_Mg2, n_Si2, n_Ca2):
    """
    N I, O I, Mg II, Si II, Ca II.  Port of wittmann.LUKEOP. Active T<30000 K.

    Pass n_Ca2 = 0.0 when Ca II is an active NLTE atom, so its bound-free edges are not
    counted both here and in get_RT_coefficients (audit F-017).
    """
    out = np.zeros_like(freq)
    # N I
    C1130 = 6.0*np.exp(-3.575/TKEV); C1020 = 10.0*np.exp(-2.384/TKEV)
    N1 = np.zeros_like(freq)
    m = freq >= 3.517915e15; N1[m] += _seaton(3.517915e15, 1.142e-17, 2.0, 4.29,  freq[m]) * 4.0
    m = freq >= 2.941534e15; N1[m] += _seaton(2.941534e15, 4.410e-18, 1.5, 3.85,  freq[m]) * C1020
    m = freq >= 2.653317e15; N1[m] += _seaton(2.653317e15, 4.200e-18, 1.5, 4.34,  freq[m]) * C1130
    out += N1 * n_N1
    # O I
    out += np.where(freq >= 3.28805e15,
                    9.0*_seaton(3.28805e15, 2.94e-18, 1.0, 2.66, freq)*n_O1, 0.0)
    # Mg II
    C1169 = 6.0*np.exp(-4.43/TKEV)
    Mg2 = np.zeros_like(freq)
    m = freq >= 3.635492e15; Mg2[m] += _seaton(3.635492e15, 1.40e-19, 4.0, 6.7, freq[m]) * 2.0
    m = freq >= 2.564306e15; Mg2[m] += 5.11e-19*(2.564306e15/freq[m])**3 * C1169
    out += Mg2 * n_Mg2
    # Si II (Peach table)
    Si2 = np.array([_si2op(f, flg, T, TLOG) for f, flg in zip(freq, FREQLG)])
    out += Si2 * n_Si2
    # Ca II
    C1218 = 10.0*np.exp(-1.697/TKEV); C1420 = 6.0*np.exp(-3.142/TKEV)
    Ca2 = np.zeros_like(freq)
    m = freq >= 2.870454e15; Ca2[m] += 1.08e-19*(2.870454e15/freq[m])**3
    m = freq >= 2.460127e15; Ca2[m] += 1.64e-17*np.sqrt(2.460127e15/freq[m])*C1218
    m = freq >= 2.110779e15; Ca2[m] += _seaton(2.110779e15, 4.13e-18, 3.0, 0.69, freq[m])*C1420
    out += Ca2 * n_Ca2
    return out * STIM


def add_background_opacity_old(iz: int, 
                            freq_grid: np.ndarray, 
                            atoms: List[MultiLevelAtom], 
                            atmosphere: Atmosphere) -> Tuple[np.ndarray, np.ndarray]:
    
    emis_c = np.zeros_like(freq_grid)
    abs_c = np.zeros_like(freq_grid)

    # B_nu and related terms
    B_nu = plank(freq_grid, atmosphere.temp[iz])

    # Retrieve the true neutral Hydrogen population (NLTE if H is active, LTE background otherwise)
    h_atom = next((a for a in atoms if a.name == "H"), None)
    if h_atom is not None:
        neutral_indices = [i for i, lvl in enumerate(h_atom.levels) if lvl.ionization == 0]
        n_H_I = np.sum(h_atom.populations[iz, neutral_indices])
    else:
        n_H_I = atmosphere.bg_species['n_HI_per_U'][iz] * 2.0
    
    n_H_I = atmosphere.nh[iz]
    
    if n_H_I > 0.0:
        # H- bound-free (John 1989 fit, includes stimulated emission)
        # Returns kappa_bf / n_H
        abs_h_bf_per_H_minus = opac_h_minus_bf_john1989(freq_grid, atmosphere.temp[iz], atmosphere.ne[iz])
        abs_h_bf = abs_h_bf_per_H_minus * n_H_I
        emis_c += abs_h_bf * B_nu
        abs_c += abs_h_bf

        # H- free-free (John 1989 fit)
        # Returns alpha_ff / n_H
        abs_h_ff_per_H_minus = opac_h_minus_ff_john1989(freq_grid, atmosphere.temp[iz], atmosphere.ne[iz])
        abs_h_ff = abs_h_ff_per_H_minus * n_H_I
        emis_c += abs_h_ff * B_nu
        abs_c += abs_h_ff
        
        # Neutral Hydrogen (Bound-Free & Bound-Bound)
        abs_h_hydro_per_H = opac_hydrogen_landi_1976(freq_grid, atmosphere.temp[iz])
        abs_h_hydro = abs_h_hydro_per_H * n_H_I
        emis_c += abs_h_hydro * B_nu
        abs_c += abs_h_hydro

        # Rayleigh scattering (H I)
        # Ported from cont_opacity.f90 (Dalgarno 1962 fit)
        sigma_rayleigh = opac_rayleigh_h_dalgarno(freq_grid)
        kappa_rayleigh = n_H_I * sigma_rayleigh
        # TO CHECK: Difference between J_nu and B?
        emis_c += kappa_rayleigh * atmosphere.J_nu[iz, :]
        abs_c += kappa_rayleigh
    
    # --- Scattering (Thomson & Rayleigh) ---
    # Thomson scattering (electrons)
    kappa_thomson = atmosphere.ne[iz] * (8*np.pi/3)*((q_e_CGS/c_CGS)**4)/m_e_CGS**2
    # TO CHECK: Difference between J_nu and B?
    emis_c += kappa_thomson * atmosphere.J_nu[iz, :]
    abs_c += kappa_thomson

    return emis_c, abs_c

def opac_h_minus_bf_john1989(freq: np.ndarray, T: float, n_e: float) -> np.ndarray:
    """
    Port of OPAC_HMINUS_BF from cont_opacity.f90 (John 1989).
    Calculates H- bound-free opacity per H atom [cm^2],
    including stimulated emission.
    kappa_bf(nu) / n_HI
    """
    lambda_A = (c_CGS / freq) * 1e8
    lambda_mic = lambda_A / 1e4

    opacity_per_HI = np.zeros_like(lambda_mic)
    
    # Constants from OPAC_HMINUS_BF
    LAMBDAP = 1.6419  # microns (16419 A)
    valid_lambda = lambda_mic[lambda_mic < LAMBDAP]

    CTE = 0.75e-18

    ALPHA = (h_CGS * c_CGS / kB_CGS) * 1e4 # h*c/k in (K * micron)
    CC = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])
   
    com_l = (1.0 / valid_lambda) - (1.0 / LAMBDAP)
    # Cross-section per H- ion
    # SIGMA = CC(1) + CC(2)*COM**0.5D0 + CC(3)*COM + CC(4)*COM**1.5D0 + CC(5)*COM**2D0 + CC(6)*COM**2.5D0
    sigma = ( CC[0] +
              CC[1] * com_l**0.5 +
              CC[2] * com_l +
              CC[3] * com_l**1.5 +
              CC[4] * com_l**2.0 +
              CC[5] * com_l**2.5 )
    # SIGMA = CTE*SIGMA*LAMBDA0MIC**3D0*COM**1.5D0
    sigma = CTE * sigma * (valid_lambda**3) * (com_l**1.5)

    # PART = T**(-2.5D0)*DEXP(ALPHA/(T*LAMBDAP))*(1D0-DEXP(-ALPHA/(T*LAMBDA0MIC)))
    part =  T**(-2.5)*np.exp(ALPHA/(T*LAMBDAP))*(1.0-np.exp(-ALPHA/(T*valid_lambda)))
    
    # Opacity per H atom (kappa_bf / n_HI)
    # OPAC = PART*SIGMA*NE*KBOL*T
    P_e = n_e * kB_CGS * T
    opacity_per_HI[lambda_mic < LAMBDAP] = part * sigma * P_e

    if not np.any(lambda_mic < LAMBDAP):
        print("Warning: No wavelengths to compute H- bf opacity, set to 0.")
    
    # SAFEGUARD: The John (1989) polynomial fit diverges heavily in the EUV. 
    # Clamp opacities below 1250 Å to zero to prevent matrix explosions.
    opacity_per_HI[lambda_A < 1250.0] = 0.0
    
    # Absolute floor to guarantee no negative cross-sections are returned
    opacity_per_HI = np.maximum(opacity_per_HI, 0.0)
    
    return opacity_per_HI

def opac_h_minus_ff_john1989(freq: np.ndarray, T: float, n_e: float) -> np.ndarray:
    """
    Port of OPAC_HMINUS_FF from cont_opacity.f90 (John 1989).
    Calculates H- free-free opacity per H atom [cm^2].
    alpha_ff(nu) / n_HI
    """
    lambda_A = (c_CGS / freq) * 1e8
    lambda_mic = lambda_A / 1e4
    theta = 5040.0 / T
    
    # Coefficients from OPAC_HMINUS_FF
    A1 = np.array([0.0, 2483.346, -3449.889, 2200.04, -696.271, 88.283])
    B1 = np.array([0.0, 285.827, -1158.382, 2427.719, -1841.4, 444.517])
    C1 = np.array([0.0, -2054.291, 8746.523, -13651.105, 8624.97, -1863.864])
    D1 = np.array([0.0, 2827.776, -11485.632, 16755.524, -10051.53, 2095.288])
    E1 = np.array([0.0, -1341.537, 5303.609, -7510.494, 4400.067, -901.788])
    F1 = np.array([0.0, 208.952, -812.939, 1132.738, -655.02, 132.985])
    
    A2 = np.array([518.1021, 473.2636, -482.2089, 115.5291])
    B2 = np.array([-734.8666, 1443.4137, -737.1616, 169.6374])
    C2 = np.array([1021.1775, -1977.3395, 1096.8827, -245.649])
    D2 = np.array([-479.0721, 922.3575, -521.1341, 114.243])
    E2 = np.array([93.1373, -178.9275, 101.7963, -21.9972])
    F2 = np.array([-6.4285, 12.36, -7.0571, 1.5097])

    part1 = np.zeros_like(lambda_mic)

    # Branch lambda_mic < 0.3645
    if np.any(lambda_mic < 0.3645):
        l = lambda_mic[lambda_mic < 0.3645]

        com2 = ( A2[np.newaxis, :] * (l**2)[:, np.newaxis] +
                 B2[np.newaxis, :] +
                 C2[np.newaxis, :] / (l)[:, np.newaxis] +
                 D2[np.newaxis, :] / (l**2)[:, np.newaxis] +
                 E2[np.newaxis, :] / (l**3)[:, np.newaxis] +
                 F2[np.newaxis, :] / (l**4)[:, np.newaxis] )
        
        theta_pows = np.array([theta**p for p in [1.0, 1.5, 2.0, 2.5]])
        part1[lambda_mic < 0.3645] = np.dot(com2, theta_pows)

    if np.any(lambda_mic >= 0.3645):
        l = lambda_mic[lambda_mic >= 0.3645]
        
        com1 = ( A1[np.newaxis, :] * (l**2)[:, np.newaxis] +
                 B1[np.newaxis, :] +
                 C1[np.newaxis, :] / (l)[:, np.newaxis] +
                 D1[np.newaxis, :] / (l**2)[:, np.newaxis] +
                 E1[np.newaxis, :] / (l**3)[:, np.newaxis] +
                 F1[np.newaxis, :] / (l**4)[:, np.newaxis] )
        
        theta_pows = np.array([theta**p for p in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]])
        part1[lambda_mic >= 0.3645] = np.dot(com1, theta_pows)  
    
    # Opacity per H atom (alpha_ff / n_HI)
    # The Fortran code calculates: 1e-29 * PART1 * (KBOL * NE * T)
    # This is (alpha_ff / (n_HI * P_e)) * P_e = alpha_ff / n_HI
    P_e = n_e * kB_CGS * T
    opacity_per_HI = 1e-29 * part1 * P_e

    # if np.any(lambda_A < 1800.0):
        # raise ValueError("Wavelengths for H- opacities should be > 1800 Amstrongs")

    # SAFEGUARD: This perfectly replicates the Fortran line 20: 
    # IF (LAMBDA0.LT.1800D0) THEN ... STOP
    opacity_per_HI[lambda_A < 1800.0] = 0.0 
    
    # Absolute floor 
    opacity_per_HI = np.maximum(opacity_per_HI, 0.0)
    
    return opacity_per_HI

def opac_rayleigh_h_dalgarno(freq: np.ndarray) -> np.ndarray:
    """
    Port of OPAC_RAYLEIGH_H from cont_opacity.f90 (Dalgarno 1962).
    Calculates Rayleigh scattering cross-section per H atom [cm^2].
    """
    lambdaA = (c_CGS / freq) * 1e8

    # Coefficients from OPAC_RAYLEIGH_H
    CC = np.array([5.799e-13, 1.422e-6, 2.784])
    
    # OPAC = (CC(1)+(CC(2)+CC(3)/LAMBDA0**2D0) / LAMBDA0**2D0) / LAMBDA0**4D0   
    sigma =  (CC[0]+(CC[1]+CC[2]/lambdaA**2.0) /  lambdaA**2.0) / lambdaA**4.0

    return sigma

def opac_hydrogen_landi_1976(freq: np.ndarray, T: float) -> np.ndarray:
    """
    Port of OPAC_HYDROGEN from cont_opacity.f90 (Landi Degl'Innocenti 1976).
    Calculates bound-free and bound-bound continuum opacity per neutral Hydrogen atom [cm^2].
    Assumes partition function of 2 (valid for T < 12000 K).
    """
    lambda_A = (c_CGS / freq) * 1e8

    # Constants directly from Fortran implementation to ensure an exact match
    C1 = 1.09651067903121578e-03  # 1e-8 * 13.595 * eV / (h * c)
    C2 = 1.04490915480325020e-26  # Pre-factor for Kramer's formula
    C3 = 143886436.07434255       # (h * c) / (k * 1e-8)
    C4 = 157773.01682218799       # 13.595 * eV / k

    # Minimum principal quantum number that can be photoionized at this wavelength
    n0 = 1 + np.floor(np.sqrt(C1 * lambda_A)).astype(int)
    
    sum_term = np.zeros_like(lambda_A)
    
    # ---------------------------------------------------------
    # Branch 1: Wavelengths ionizing n0 <= 8
    # ---------------------------------------------------------
    mask_le8 = n0 <= 8
    if np.any(mask_le8):
        # Explicit sum from local n0 up to 8
        for n in range(1, 9):
            valid_n = (n >= n0) & mask_le8
            sum_term[valid_n] += np.exp(C4 / (T * n**2)) * (n**-3)
            
        # Add high-N analytical approximation (n=9 to infinity)
        sum_term[mask_le8] += (0.117 + np.exp(C4 / (T * 9.0**2))) * (T / (2.0 * C4))

    # ---------------------------------------------------------
    # Branch 2: Wavelengths ionizing n0 > 8
    # ---------------------------------------------------------
    mask_gt8 = ~mask_le8
    if np.any(mask_gt8):
        # Use only the high-N analytical approximation starting from n0
        sum_term[mask_gt8] = (0.117 + np.exp(C4 / (T * n0[mask_gt8]**2))) * (T / (2.0 * C4))

    # Final cross-section calculation
    opac = C2 * sum_term * (1.0 - np.exp(-C3 / (T * lambda_A))) * np.exp(-C4 / T) * (lambda_A**3)
    
    return np.maximum(opac, 0.0)

# --------------------------------------------------------------------------
# Formal solution with linear Short Characteristics and MALI
def formal_solution(ray, I_m, dz, emis_M, emis_O, abs_M, abs_O):

    delta_tauMO = 0.5*(abs_M + abs_O)*np.abs(dz/ray) + vacuum_CGS
    exp_tauMO = np.exp(-delta_tauMO)

    # Guard against zero absorption (vacuum points) — use emis directly
    # abs_M_safe = np.where(abs_M > vacuum_CGS, abs_M, vacuum_CGS)
    # abs_O_safe = np.where(abs_O > vacuum_CGS, abs_O, vacuum_CGS)
    S_m = emis_M / abs_M #_safe
    S_o = emis_O / abs_O #_safe

    if np.any(S_m < 0) or np.any(S_o < 0):
        print("Warning: Negative source function encountered. Check emissivities and absorptions.")

    # Small linear
    small = (delta_tauMO < 1e-7)
    exp_tauMO[small] = 1. - delta_tauMO[small] + 0.5*delta_tauMO[small]*delta_tauMO[small]

    psi_m, psi_o = psi_lin(exp_tauMO, delta_tauMO)
    I_o = I_m*exp_tauMO + psi_m*S_m + psi_o*S_o

    # The local diagonal operator for this step is exactly psi_o
    # (Since S_o = emis_O / abs_O, the derivative dI_o / dS_o is psi_o)
    Lambda_star_mu = psi_o

    return I_o, Lambda_star_mu

def psi_lin(exp_dtau,dtau):
    """
    Compute linear contributions
    COPIED FROM ORIGINAL HE 1083 CODE
    """

    big = dtau > 0.10
    small = dtau <= 0.10

    psi_m = np.empty(dtau.shape)
    psi_o = np.empty(dtau.shape)

    psi_m[small] = ((dtau[small]*(dtau[small]*(dtau[small]*(dtau[small]*(dtau[small]*(dtau[small]* \
                    ((63e0 - 8e0*dtau[small])*dtau[small] - 432e0) + 2520e0) - \
                    12096e0) + 45360e0) - 120960e0) + 181440e0))/362880e0)
    psi_o[small] = ((dtau[small]*(dtau[small]*(dtau[small]*(dtau[small]*(dtau[small]*(dtau[small]* \
                   ((9e0 - dtau[small])*dtau[small] - 72e0) + 504e0) - \
                    3024e0) + 15120e0) - 60480e0) + 181440e0))/362880e0)

    psi_m[big] = (1.-exp_dtau[big]*(1.+dtau[big]))/dtau[big]
    psi_o[big] = (exp_dtau[big]+dtau[big]-1.)/dtau[big]
   #psi_m = (1.-exp_dtau*(1.+dtau))/dtau
   #psi_o = (exp_dtau+dtau-1.)/dtau

    return psi_m,psi_o