import numpy as np
from constants import *


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

def formal_solution(ray, I_m, dz, emis_M, emis_O, abs_M, abs_O):

    delta_tauMO = 0.5*(abs_M + abs_O)*np.abs(dz/ray) + vacuum_CGS
    exp_tauMO = np.exp(-delta_tauMO)

    S_m = emis_M / abs_M
    S_o = emis_O / abs_O

    # Small linear
    small = (delta_tauMO < 1e-7)
    exp_tauMO[small] = 1. - delta_tauMO[small] + 0.5*delta_tauMO[small]*delta_tauMO[small]

    psi_m, psi_o = psi_lin(exp_tauMO, delta_tauMO)
    I_o = I_m*exp_tauMO + psi_m*S_m + psi_o*S_o
    #     I_m*np.exp(-delta_tauM) + \
    #     (S_o - S_m)/delta_tauM * (delta_tauM - (1 - np.exp(-delta_tauM))) + \
    #     S_m * (1 - np.exp(-delta_tauM))

    # return I_o

    # The local diagonal operator for this step is exactly psi_o
    # (Since S_o = emis_O / abs_O, the derivative dI_o / dS_o is psi_o)
    Lambda_star_mu = np.zeros_like(psi_o)
    valid = abs_O > vacuum_CGS
    Lambda_star_mu[valid] = psi_o[valid]

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