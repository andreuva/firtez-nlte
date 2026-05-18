import numpy as np
from constants import *

from typing import TYPE_CHECKING, List, Dict
from atoms import MultiLevelAtom
from atmosphere import Atmosphere

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
    # eta_c, kappa_c = add_background_opacity(iz, freq_grid, atoms, atmosphere)
    # emis += eta_c
    # abs += kappa_c
    
    # Sanity check for negative absorptions or emissivities
    if np.any(emis < 0):
        print(f"Warning: Negative emission at depth {iz}.")
        # emis = np.maximum(emis, 0)
    if np.any(abs < 0):
        print(f"Warning: Negative absorption at depth {iz}.")
        # abs = np.maximum(abs, vacuum_CGS)

    return emis, abs

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