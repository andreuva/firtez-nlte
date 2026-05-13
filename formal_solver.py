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


