import numpy as np
from constants import *
from typing import TYPE_CHECKING, List, Dict, Tuple

from atoms import MultiLevelAtom
from atmosphere import Atmosphere
from chemeq import XII

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
            
            total_damping = 0.0
            dop_freq = (freq_grid - line.nu0)/atom.doppler_widths[iz, il]

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

            n_u = atom.populations[iz, line.upper_level_index]
            n_l = atom.populations[iz, line.lower_level_index]

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
    
    emis_c = np.zeros_like(freq_grid)
    abs_c = np.zeros_like(freq_grid)

    # B_nu and related terms
    B_nu = plank(freq_grid, atmosphere.temp[iz])
    n_H_I = atmosphere.nh[iz]
    
    if n_H_I > 0.0:
        # H- bound-free (John 1989 fit, includes stimulated emission)
        # Returns kappa_bf / n_H
        abs_h_bf_per_H_minus = opac_h_minus_bf_john1989(freq_grid, atmosphere.temp[iz], atmosphere.ne[iz])
        abs_h_bf = abs_h_bf_per_H_minus * n_H_I
        emis_h_bf = abs_h_bf * B_nu # Assumes S_nu(H-) = B_nu
        
        emis_c += emis_h_bf
        abs_c += abs_h_bf

        # H- free-free (John 1989 fit)
        # Returns alpha_ff / n_H
        abs_h_ff_per_H_minus = opac_h_minus_ff_john1989(freq_grid, atmosphere.temp[iz], atmosphere.ne[iz])
        abs_h_ff = abs_h_ff_per_H_minus * n_H_I
        emis__h_ff = abs_h_ff * B_nu # Assumes S_nu(H-) = B_nu
        
        emis_c += emis__h_ff
        abs_c += abs_h_ff
    
    # --- Scattering (Thomson & Rayleigh) ---
    # Thomson scattering (electrons)
    kappa_thomson = atmosphere.ne[iz] * (8*np.pi/3)*((q_e_CGS/c_CGS)**4)/m_e_CGS**2
    emis_c += kappa_thomson * B_nu #* J_nu
    abs_c += kappa_thomson
    
    # Rayleigh scattering (H I)
    # Ported from cont_opacity.f90 (Dalgarno 1962 fit)
    if n_H_I > 0.0:
        sigma_rayleigh = opac_rayleigh_h_dalgarno(freq_grid)
        kappa_rayleigh = n_H_I * sigma_rayleigh
        
        emis_c += kappa_rayleigh * B_nu #* J_nu
        abs_c += kappa_rayleigh

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

    opacity_per_HI[lambda_A < 1800.0] = 0.0 # From Fortran check
    
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

# --------------------------------------------------------------------------
# Formal solution with linear Short Characteristics and MALI
def formal_solution(ray, I_m, dz, emis_M, emis_O, abs_M, abs_O):

    delta_tauMO = 0.5*(abs_M + abs_O)*np.abs(dz/ray) + vacuum_CGS
    exp_tauMO = np.exp(-delta_tauMO)

    S_m = emis_M / abs_M
    S_o = emis_O / abs_O

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