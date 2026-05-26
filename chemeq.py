import numpy as np
from constants import *
from atoms import MultiLevelAtom

# =====================================================================
# ATOMIC DATABASE
# =====================================================================
NELEM = 92
# First ionization potential (For H, this is the binding energy of H-) [eV]
XI = np.array([
    0.754, 24.58, 5.39, 9.32, 8.298, 11.256, 14.529, 13.614, 17.418, 21.559, 
    5.138, 7.644, 5.984, 8.149, 10.474, 10.357, 13.012, 15.755, 4.339, 6.111, 
    6.538, 6.825, 6.738, 6.763, 7.432, 7.896, 7.863, 7.633, 7.724, 9.391, 
    5.997, 7.88, 9.81, 9.75, 11.840, 13.996, 4.176, 5.692, 6.377, 6.838, 
    6.881, 7.10, 7.28, 7.36, 7.46, 8.33, 7.574, 8.991, 5.785, 7.34, 
    8.64, 9.01, 10.454, 12.127, 3.893, 5.210, 5.577, 5.466, 5.422, 5.489, 
    5.554, 5.631, 5.666, 6.141, 5.852, 5.927, 6.018, 6.101, 6.184, 6.254, 
    5.426, 6.650, 7.879, 7.980, 7.870, 8.70, 9.10, 9.00, 9.22, 10.43, 
    6.105, 7.415, 7.287, 8.43, 9.30, 10.745, 4.0, 5.276, 6.9, 6.0, 6.0, 6.0
])

# Second ionization potential (For H, this is the standard first ionization) [eV]
XII = np.array([
    13.595, 54.403, 75.62, 18.21, 25.15, 24.376, 29.59, 35.11, 34.98, 41.07,
    47.290, 15.03, 18.823, 16.34, 19.72, 23.405, 23.798, 27.62, 31.81, 11.868,
    12.891, 13.63, 14.205, 16.493, 15.636, 16.178, 17.052, 18.15, 20.286, 17.96,
    20.509, 15.93, 18.63, 21.50, 21.60, 24.565, 27.50, 11.027, 12.233, 13.13,
    14.316, 16.15, 15.26, 16.76, 18.07, 19.42, 21.48, 16.904, 18.86, 14.63,
    16.50, 18.60, 19.09, 21.20, 25.10, 10.001, 11.060, 10.850, 10.550, 10.730,
    10.899, 11.069, 11.241, 12.090, 11.519, 11.670, 11.800, 11.930, 12.050,
    12.184, 13.900, 14.900, 16.2, 17.7, 16.60, 17.00, 20.00, 18.56, 20.50, 
    18.75, 20.42, 15.03, 16.68, 19.0, 20.0, 20.0, 22.0, 10.144, 12.1, 12.0, 
    12.0, 12.0
])

# Abundances (THEVENIN 1989)
ABUND = np.array([
    12.00, 11.00, 1.00, 1.15, 2.60, 8.69, 7.99, 8.91, 4.56, 8.00, 6.28,
    7.53, 6.43, 7.50, 5.45, 7.21, 5.50, 6.58, 5.05, 6.36, 2.99, 4.88, 
    3.91, 5.61, 5.47, 7.46, 4.85, 6.18, 4.24, 4.60, 2.88, 3.57, 2.39, 
    3.35, 2.63, 3.21, 2.60, 2.93, 2.18, 2.46, 1.46, 2.10, 0.00, 1.78, 
    1.10, 1.69, 0.94, 1.86, 1.66, 2.00, 1.00, 2.25, 1.51, 2.19, 1.12, 
    2.18, 1.07, 1.58, 0.76, 1.40, 0.00, 0.88, 0.48, 1.13, 0.20, 1.07, 
    0.26, 0.93, 0.00, 1.08, 0.76, 0.88, -0.09, 0.98, 0.26, 1.45, 1.36, 
    1.80, 1.13, 1.27, 0.90, 1.90, 0.71, -8.0, -8.0, -8.0, -8.0, -8.0, 
    -8.0, 0.02, -8.0, -0.47
])

# Atomic mass [amu]
MATOM = np.array([
    1.008, 4.003, 6.939, 9.012, 10.811, 12.011, 14.007, 16.0, 18.998,
    20.183, 22.99, 24.312, 26.982, 28.086, 30.974, 32.064, 35.453, 39.948,
    39.102, 40.08, 44.956, 47.90, 50.942, 51.996, 54.938, 55.847, 58.933,
    58.71, 63.54, 65.37, 69.72, 72.59, 74.92, 78.96, 79.91, 83.80, 85.47, 
    87.62, 88.905, 91.22, 92.906, 95.94, 99.00, 101.07, 102.9, 106.4, 
    107.87, 112.40, 114.82, 118.69, 121.75, 127.6, 126.9, 131.3, 132.9, 
    137.34, 138.91, 140.12, 140.91, 144.24, 147.00, 150.35, 151.96, 157.25, 
    158.92, 162.50, 164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95, 
    183.85, 186.2, 190.2, 192.2, 195.09, 196.97, 200.59, 204.37, 207.19, 
    208.98, 210.0, 211.0, 222.0, 223.0, 226.1, 227.1, 232.04, 231.0, 238.03
])

# =====================================================================
# PARTITION FUNCTIONS
# =====================================================================
def get_partition_functions(Z, T):
    """
    Returns partition functions (UI, UII, UIII) for element Z at temperature T.
    Exactly mirrors the FIRTEZ atom_database.f90 PARTITION_FUNCTION routine.
    """
    UI, UII, UIII = 0.0, 0.0, 0.0
    x = np.log(5040.0 / T)
    y = 1e-3 * T
    
    if Z == 1: # H
        UI = 1.0
        UII = 2.0
        if T > 1.3e4: UII = 1.51 + 3.8e-5 * T
        if T > 1.62e4: UII = 11.41 + T * (-1.1428e-3 + T * 3.52e-8)
        UIII = 1.0
    elif Z == 2: # He
        UI = 1.0
        if T > 3e4: UI = 14.8 + T * (-9.4103e-4 + T * 1.6095e-8)
        UII = 2.0
        UIII = 1.0
    elif Z == 3: # Li
        UI = 2.081 - y * (6.8926e-2 - y * 1.4081e-2)
        if T > 6e3: UI = 3.4864 + T * (-7.3292e-4 + T * 8.5586e-8)
        UII, UIII = 1.0, 2.0
    elif Z == 4: # Be
        UI, UII, UIII = max(1.0, 0.631 + 7.032e-5 * T), 2.0, 1.0
    elif Z == 5: # B
        UI, UII, UIII = 5.9351 + 1.0438e-2 * y, 1.0, 2.0
    elif Z == 6: # C
        UI = 8.6985 + y * (2.0485e-2 + y * (1.7629e-2 - 3.9091e-4 * y))
        if T > 1.2e4: UI = 13.97 + T * (-1.3907e-3 + T * 9.0844e-8)
        UII = 5.838 + 1.6833e-5 * T
        if T > 2.4e4: UII = 10.989 + T * (-6.9347e-4 + T * 2.0861e-8)
        UIII = 1.0
        if T > 1.95e4: UIII = -0.555 + 8e-5 * T
    elif Z == 7: # N
        UI = 3.9914 + y * (1.7491e-2 - y * (1.0148e-2 - y * 1.7138e-3))
        if T > 8800.0: UI = 2.171 + 2.54e-4 * T
        if T > 1.8e4: UI = 11.396 + T * (-1.7139e-3 + T * 8.633e-8)
        UII = 8.060 + 1.420e-4 * T
        if T > 3.3e4: UII = 26.793 + T * (-1.8931e-3 + T * 4.4612e-8)
        UIII = 5.9835 + T * (-2.6651e-5 + T * 1.8228e-9)
        if T < 7310.5: UIII = 5.89
    elif Z == 8: # O
        UI = 8.29 + 1.10e-4 * T
        if T > 1.9e4: UI = 66.81 + T * (-6.019e-3 + T * 1.657e-7)
        UII = max(4.0, 3.51 + 8e-5 * T)
        if T > 3.64e4: UII = 68.7 + T * (-4.216e-3 + T * 6.885e-8)
        UIII = 7.865 + 1.1348e-4 * T
    elif Z == 9: # F
        UI = 4.5832 + y * (0.77683 + y * (-0.20884 + y * (2.6771e-2 - 1.3035e-3 * y)))
        if T > 8750.0: UI = 5.9
        if T > 2e4: UI = 15.16 + T * (-9.229e-4 + T * 2.312e-8)
        UII, UIII = 8.15 + 8.9e-5 * T, 2.315 + 1.38e-4 * T
    elif Z == 10: # Ne
        UI = 1.0
        if T > 2.69e4: UI = 26.3 + T * (-2.113e-3 + T * 4.359e-8)
        UII, UIII = 5.4 + 4e-5 * T, 7.973 + 7.956e-5 * T
    elif Z == 11: # Na
        UI = max(2.0, 1.72 + 9.3e-5 * T)
        if T > 5400.0: UI = -0.83 + 5.66e-4 * T
        if T > 8.5e3: UI = 4.5568 + T * (-1.2415e-3 + T * 1.3861e-7)
        UII, UIII = 1.0, 5.69 + 5.69e-6 * T
    elif Z == 12: # Mg
        UI = 1.0 + np.exp(-4.027262 - x * (6.173172 + x * (2.889176 + x * (2.393895 + 0.784131 * x))))
        if T > 8e3: UI = 2.757 + T * (-7.8909e-4 + T * 7.4531e-8)
        UII = 2.0 + np.exp(-7.721172 - x * (7.600678 + x * (1.966097 + 0.212417 * x)))
        if T > 2e4: UII = 7.1041 + T * (-1.0817e-3 + T * 4.7841e-8)
        UIII = 1.0
    elif Z == 13: # Al
        UI = 5.2955 + y * (0.27833 - y * (4.7529e-2 - y * 3.0199e-3))
        UII = max(1.0, 0.725 + 3.245e-5 * T)
        if T > 2.24e4: UII = 61.06 + T * (-5.987e-3 + T * 1.485e-7)
        UIII = max(2.0, 1.976 + 3.43e-6 * T)
        if T > 1.814e4: UIII = 3.522 + T * (-1.59e-4 + T * 4.382e-9)
    elif Z == 14: # Si
        UI = 6.7868 + y * (0.86319 + y * (-0.11622 + y * (0.013109 - 6.2013e-4 * y)))
        if T > 1.04e4: UI = 86.01 + T * (-1.465e-2 + T * 7.282e-7)
        UII = 5.470 + 4e-5 * T
        if T > 1.8e4: UII = 26.44 + T * (-2.22e-3 + T * 6.188e-8)
        UIII = max(1.0, 0.911 + 1.1e-5 * T)
        if T > 3.33e4: UIII = 19.14 + T * (-1.408e-3 + T * 2.617e-8)
    elif Z == 15: # P
        UI = 4.2251 + y * (-0.22476 + y * (0.057306 - y * 1.0381e-3))
        if T > 6e3: UI = 1.56 + 5.2e-4 * T
        UII = 4.4151 + y * (2.2494 + y * (-0.55371 + y * (0.071913 - y * 3.5156e-3)))
        if T > 7250.0: UII = 4.62 + 5.38e-4 * T
        UIII = 5.595 + 3.4e-5 * T
    elif Z == 16: # S
        UI = 7.5 + 2.15e-4 * T
        if T > 1.16e4: UI = 38.76 + T * (-4.906e-3 + T * 2.125e-7)
        UII = 2.845 + 2.43e-4 * T
        if T > 1.05e4: UII = 6.406 + T * (-1.68e-4 + T * 1.323e-8)
        UIII = 7.38 + 1.88e-4 * T
    elif Z == 17: # Cl
        UI = 5.2 + 6e-5 * T
        if T > 1.84e4: UI = -81.6 + 4.8e-3 * T
        UII, UIII = 7.0 + 2.43e-4 * T, 2.2 + 2.62e-4 * T
    elif Z == 18: # Ar
        UI, UII, UIII = 1.0, 5.20 + 3.8e-5 * T, 7.474 + 1.554e-4 * T
    elif Z == 19: # K
        UI = 1.9909 + y * (0.023169 - y * (0.017432 - y * 4.0938e-3))
        if T > 5800.0: UI = -9.93 + 2.124e-3 * T
        UII, UIII = 1.0, 5.304 + 1.93e-5 * T
    elif Z == 20: # Ca
        UI = 1.0 + np.exp(-1.731273 - x * (5.004556 + x * (1.645456 + x * (1.326861 + 0.508553 * x))))
        UII = 2.0 + np.exp(-1.582112 - x * (3.996089 + x * (1.890737 + 0.539672 * x)))
        UIII = 1.0
    elif Z == 21: # Sc
        UI = 4.0 + np.exp(2.071563 + x * (-1.2392 + x * (1.173504 + 0.517796 * x)))
        UII = 3.0 + np.exp(2.988362 + x * (-0.596238 + 0.054658 * x))
        UIII = 10.0
    elif Z == 22: # Ti
        UI = 5.0 + np.exp(3.200453 + x * (-1.227798 + x * (0.799613 + 0.278963 * x)))
        if T < 5.5e3: UI = 16.37 + T * (-2.838e-4 + T * 5.819e-7)
        UII = 4.0 + np.exp(3.94529 + x * (-0.551431 + 0.115693 * x))
        UIII = 16.4 + 8.5e-4 * T
    elif Z == 23: # V
        UI = 4.0 + np.exp(3.769611 + x * (-0.906352 + x * (0.724694 + 0.1622 * x)))
        UII = 1.0 + np.exp(3.755917 + x * (-0.757371 + 0.21043 * x))
        UIII = -18.0 + 1.03e-2 * T
        if T < 2.25e3: UIII = 2.4e-3 * T
    elif Z == 24: # Cr
        UI = 7.0 + np.exp(1.225042 + x * (-2.923459 + x * (0.154709 + 0.09527 * x)))
        UII = 6.0 + np.exp(0.128752 - x * (4.143973 + x * (1.096548 + 0.230073 * x)))
        UIII = 10.4 + 2.1e-3 * T
    elif Z == 25: # Mn
        UI = 6.0 + np.exp(-0.86963 - x * (5.531252 + x * (2.13632 + x * (1.061055 + 0.265557 * x))))
        UII = 7.0 + np.exp(-0.282961 - x * (3.77279 + x * (0.814675 + 0.159822 * x)))
        UIII = 10.0
    elif Z == 26: # Fe
        UI = 9.0 + np.exp(2.930047 + x * (-0.979745 + x * (0.76027 + 0.118218 * x)))
        if T < 4e3: UI = 15.85 + T * (1.306e-3 + T * 2.04e-7)
        if T > 9e3: UI = 39.149 + T * (-9.5922e-3 + T * 1.2477e-6)
        UII = 10.0 + np.exp(3.501597 + x * (-0.612094 + 0.280982 * x))
        if T > 1.8e4: UII = 68.356 + T * (-6.1104e-3 + T * 5.1567e-7)
        UIII = 17.336 + T * (5.5048e-4 + T * 5.7514e-8)
    elif Z == 27: # Co
        UI, UII, UIII = 8.65 + 4.9e-3 * T, 11.2 + 3.58e-3 * T, 15.0 + 1.42e-3 * T
    elif Z == 28: # Ni
        UI = 9.0 + np.exp(3.084552 + x * (-0.401323 + x * (0.077498 - 0.278468 * x)))
        UII = 6.0 + np.exp(1.593047 - x * (1.528966 + 0.115654 * x))
        UIII = 13.3 + 6.9e-4 * T
    elif Z == 29: # Cu
        UI = max(2.0, 1.50 + 1.51e-4 * T)
        if T > 6250.0: UI = -0.3 + 4.58e-4 * T
        UII, UIII = max(1.0, 0.22 + 1.49e-4 * T), 8.025 + 9.4e-5 * T
    elif Z == 30: # Zn
        UI, UII, UIII = max(1.0, 0.632 + 5.11e-5 * T), 2.0, 1.0
    elif Z == 31: # Ga
        UI = 1.7931 + y * (1.9338 + y * (-0.4643 + y * (0.054876 - y * 2.5054e-3)))
        if T > 6e3: UI = 4.18 + 2.03e-4 * T
        UII, UIII = 1.0, 2.0
    elif Z == 32: # Ge
        UI, UII, UIII = 6.12 + 4.08e-4 * T, 3.445 + 1.78e-4 * T, 1.1
    elif Z == 33: # As
        UI = 2.65 + 3.65e-4 * T
        UII = -0.25384 + y * (2.284 + y * (-0.33383 + y * (0.030408 - y * 1.1609e-3)))
        if T > 1.2e4: UII = 8.0
        UIII = 8.0
    elif Z == 34: # Se
        UI, UII, UIII = 6.34 + 1.71e-4 * T, 4.1786 + y * (-0.15392 + 3.2053e-2 * y), 8.0
    elif Z == 35: # Br
        UI, UII, UIII = 4.12 + 1.12e-4 * T, 5.22 + 3.08e-4 * T, 2.3 + 2.86e-4 * T
    elif Z == 36: # Kr
        UI, UII, UIII = 1.0, 4.11 + 7.4e-5 * T, 5.35 + 2.23e-4 * T
    elif Z == 37: # Rb
        UI = max(2.0, 1.38 + 1.94e-4 * T)
        if T > 6250.0: UI = -14.9 + 2.79e-3 * T
        UII, UIII = 1.0, 4.207 + 4.85e-5 * T
    elif Z == 38: # Sr
        UI = 0.87127 + y * (0.20148 + y * (-0.10746 + y * (0.021424 - y * 1.0231e-3)))
        if T > 6500.0: UI = -6.12 + 1.224e-3 * T
        UII, UIII = max(2.0, 0.84 + 2.6e-4 * T), 1.0
    elif Z == 39: # Y
        UI, UII, UIII = 0.2 + 2.58e-3 * T, 7.15 + 1.855e-3 * T, 9.71 + 9.9e-5 * T
    elif Z == 40: # Zr
        UI = 76.31 + T * (-1.866e-2 + T * 2.199e-6)
        if T < 6236.0: UI = 6.8 + T * (2.806e-3 + T * 5.386e-7)
        UII, UIII = 4.0 + np.exp(3.721329 - 0.906502 * x), 12.3 + 1.385e-3 * T
    elif Z == 41: # Nb
        UI, UII, UIII = max(1.0, -19.0 + 1.43e-2 * T), -4.0 + 1.015e-2 * T, 25.0
    elif Z == 42: # Mo
        UI = max(7.0, 2.1 + 1.5e-3 * T)
        if T > 7e3: UI = -38.1 + 7.28e-3 * T
        UII = 1.25 + 1.17e-3 * T
        if T > 6900.0: UII = -28.5 + 5.48e-3 * T
        UIII = 24.04 + 1.464e-4 * T
    elif Z == 43: # Tc
        UI = 4.439 + y * (0.30648 + y * (1.6525 + y * (-0.4078 + y * (0.048401 - y * 2.1538e-3))))
        if T > 6e3: UI = 24.0
        UII = 8.1096 + y * (-2.963 + y * (2.369 + y * (-0.502 + y * (0.049656 - y * 1.9087e-3))))
        if T > 6e3: UII = 17.0
        UIII = 220.0
    elif Z == 44: # Ru
        UI, UII, UIII = -3.0 + 7.17e-3 * T, 3.0 + 4.26e-3 * T, 22.0
    elif Z == 45: # Rh
        UI = 6.9164 + y * (3.8468 + y * (0.043125 - y * (8.7907e-3 - y * 5.9589e-4)))
        UII = 7.2902 + y * (1.7476 + y * (-0.038257 + y * (2.014e-3 + y * 2.1218e-4)))
        UIII = 30.0
    elif Z == 46: # Pd
        UI, UII, UIII = max(1.0, -1.75 + 9.86e-4 * T), 5.60 + 3.62e-4 * T, 20.0
    elif Z == 47: # Ag
        UI, UII, UIII = max(2.0, 1.537 + 7.88e-5 * T), max(1.0, 0.73 + 3.4e-5 * T), 6.773 + 1.248e-4 * T
    elif Z == 48: # Cd
        UI, UII, UIII = max(1.0, 0.43 + 7.6e-5 * T), 2.0, 1.0
    elif Z == 49: # In
        UI, UII, UIII = 2.16 + 3.92e-4 * T, 1.0, 2.0
    elif Z == 50: # Sn
        UI, UII, UIII = 2.14 + 6.16e-4 * T, 2.06 + 2.27e-4 * T, 1.05
    elif Z == 51: # Sb
        UI, UII, UIII = 2.34 + 4.86e-4 * T, 0.69 + 5.36e-4 * T, 3.5
    elif Z == 52: # Te
        UI = 3.948 + 4.56e-4 * T
        UII = 4.2555 + y * (-0.25894 + y * (0.06939 - y * 2.4271e-3))
        if T > 1.2e4: UII = 7.0
        UIII = 5.0
    elif Z == 53: # I
        UI, UII, UIII = max(4.0, 3.8 + 9.5e-5 * T), 4.12 + 3e-4 * T, 7.0
    elif Z == 54: # Xe
        UI, UII, UIII = 1.0, 3.75 + 6.876e-5 * T, 4.121 + 2.323e-4 * T
    elif Z == 55: # Cs
        UI = max(2.0, 1.56 + 1.67e-4 * T)
        if T > 4850.0: UI = -2.680 + 1.04e-3 * T
        UII, UIII = 1.0, 3.769 + 4.971e-5 * T
    elif Z == 56: # Ba
        UI = max(1.0, -1.8 + 9.85e-4 * T)
        if T > 6850.0: UI = -16.2 + 3.08e-3 * T
        UII, UIII = 1.11 + 5.94e-4 * T, 1.0
    elif Z == 57: # La
        UI = 15.42 + 9.5e-4 * T
        if T > 5060.0: UI = 1.0 + 3.8e-3 * T
        UII, UIII = 13.2 + 3.56e-3 * T, 12.0
    elif Z == 58: # Ce
        UI = 9.0 + np.exp(5.202903 + x * (-1.98399 + x * (0.119673 + 0.179675 * x)))
        UII = 8.0 + np.exp(5.634882 - x * (1.459196 + x * (0.310515 + 0.052221 * x)))
        UIII = 9.0 + np.exp(3.629123 - x * (1.340945 + x * (0.372409 + x * (0.03186 - 0.014676 * x))))
    elif Z == 59: # Pr
        UII = 9.0 + np.exp(4.32396 - x * (1.191467 + x * (0.149498 + 0.028999 * x)))
        UI = UII
        UIII = 10.0 + np.exp(3.206855 + x * (-1.614554 + x * (0.489574 + 0.277916 * x)))
    elif Z == 60: # Nd
        UI = 9.0 + np.exp(4.456882 + x * (-2.779176 + x * (0.082258 + x * (0.50666 + 0.127326 * x))))
        UII = 8.0 + np.exp(4.689643 + x * (-2.039946 + x * (0.17193 + x * (0.26392 + 0.038225 * x))))
        UIII = UII
    elif Z == 61: # Pm
        UI, UII, UIII = 20.0, 25.0, 100.0
    elif Z == 62: # Sm
        UI = 1.0 + np.exp(3.549595 + x * (-1.851549 + x * (0.9964 + 0.566263 * x)))
        UII = 2.0 + np.exp(4.052404 + x * (-1.418222 + x * (0.358695 + 0.161944 * x)))
        UIII = 1.0 + np.exp(3.222807 - x * (0.699473 + x * (-0.056205 + x * (0.533833 + 0.251011 * x))))
    elif Z == 63: # Eu
        UI = 8.0 + np.exp(1.024374 - x * (4.533653 + x * (1.540805 + x * (0.827789 + 0.286737 * x))))
        UII = 9.0 + np.exp(1.92776 + x * (-1.50646 + x * (0.379584 + 0.05684 * x)))
        UIII = 8.0
    elif Z == 64: # Gd
        UI = 5.0 + np.exp(4.009587 + x * (-1.583513 + x * (0.800411 + 0.388845 * x)))
        UII = 6.0 + np.exp(4.362107 - x * (1.208124 + x * (-0.074813 + x * (0.076453 + 0.055475 * x))))
        UIII = 5.0 + np.exp(3.412951 - x * (0.50271 + x * (0.042489 - 4.017e-3 * x)))
    elif Z == 65: # Tb
        UI = 16.0 + np.exp(4.791661 + x * (-1.249355 + x * (0.570094 + 0.240203 * x)))
        UII = 15.0 + np.exp(4.472549 - x * (0.295965 + x * (5.88e-3 + 0.131631 * x)))
        UIII = UII
    elif Z == 66: # Dy
        UI = 17.0 + np.exp(3.029646 - x * (3.121036 + x * (0.086671 - 0.216214 * x)))
        UII = 18.0 + np.exp(3.465323 - x * (1.27062 + x * (-0.382265 + x * (0.431447 + 0.303575 * x))))
        UIII = UII
    elif Z == 67: # Ho
        UIII = 16.0 + np.exp(1.610084 - x * (2.373926 + x * (0.133139 - 0.071196 * x)))
        UI, UII = UIII, UIII
    elif Z == 68: # Er
        UI = 13.0 + np.exp(2.895648 - x * (2.968603 + x * (0.561515 + x * (0.215267 + 0.095813 * x))))
        UII = 14.0 + np.exp(3.202542 - x * (0.852209 + x * (-0.226622 + x * (0.343738 + 0.186042 * x))))
        UIII = UII
    elif Z == 69: # Tm
        UI = 8.0 + np.exp(1.021172 - x * (4.94757 + x * (1.081603 + 0.034811 * x)))
        UII = 9.0 + np.exp(2.173152 + x * (-1.295327 + x * (1.940395 + 0.813303 * x)))
        UIII = 8.0 + np.exp(-0.567398 + x * (-3.383369 + x * (0.799911 + 0.554397 * x)))
    elif Z == 70: # Yb
        UI = 1.0 + np.exp(-2.350549 - x * (6.688837 + x * (1.93869 + 0.269237 * x)))
        UII = 2.0 + np.exp(-3.047465 - x * (7.390444 + x * (2.355267 + 0.44757 * x)))
        UIII = 1.0 + np.exp(-6.192056 - x * (10.560552 + x * (4.579385 + 0.940171 * x)))          
    elif Z == 71: # Lu
        UI = 4.0 + np.exp(1.537094 + x * (-1.140264 + x * (0.608536 + 0.193362 * x)))
        UII = max(1.0, 0.66 + 1.52e-4 * T)
        if T > 5250.0: UII = -1.09 + 4.86e-4 * T
        UIII = 5.0
    elif Z == 72: # Hf
        UI = 4.1758 + y * (0.407 + y * (0.57862 - y * (0.072887 - y * 3.6848e-3)))
        UII, UIII = -2.979 + 3.095e-3 * T, 30.0
    elif Z == 73: # Ta
        UI = 3.0679 + y * (0.81776 + y * (0.34936 + y * (7.4861e-3 + y * 3.0739e-4)))
        UII = 1.6834 + y * (2.0103 + y * (0.56443 - y * (0.031036 - y * 8.9565e-4)))
        UIII = 15.0
    elif Z == 74: # W
        UI = 0.3951 + y * (-0.25057 + y * (1.4433 + y * (-0.34373 + y * (0.041924 - y * 1.84e-3))))
        if T > 1.2e4: UI = 23.0
        UII = 1.055 + y * (1.0396 + y * (0.3303 - y * (8.4971e-3 - y * 5.5794e-4)))
        UIII = 20.0
    elif Z == 75: # Re
        UI = 5.5671 + y * (0.72721 + y * (-0.42096 + y * (0.09075 - y * 3.9331e-3)))
        if T > 1.2e4: UI = 29.0
        UII = 6.5699 + y * (0.59999 + y * (-0.28532 + y * (0.050724 - y * 1.8544e-3)))
        if T > 1.2e4: UII = 22.0
        UIII = 20.0
    elif Z == 76: # Os
        UI = 8.6643 + y * (-0.32516 + y * (0.68181 - y * (0.044252 - y * 1.9975e-3)))
        UII = 9.7086 + y * (-0.3814 + y * (0.65292 - y * (0.064984 - y * 2.8792e-3)))
        UIII = 10.0
    elif Z == 77: # Ir
        UI = 11.07 + y * (-2.412 + y * (1.9388 + y * (-0.34389 + y * (0.033511 - 1.3376e-3 * y))))
        if T > 1.2e4: UI = 30.0
        UII, UIII = 15.0, 20.0
    elif Z == 78: # Pu
        UI = 16.4 + 1.27e-3 * T
        UII = 6.5712 + y * (-1.0363 + y * (0.57234 - y * (0.061219 - 2.6878e-3 * y)))
        UIII = 15.0
    elif Z == 79: # Au
        UI = 1.24 + 2.79e-4 * T
        UII = 1.0546 + y * (-0.040809 + y * (2.8439e-3 + y * 1.6586e-3))
        UIII = 7.0
    elif Z == 80: # Hg
        UI, UII, UIII = 1.0, 2.0, max(1.0, 0.669 + 3.976e-5 * T)
    elif Z == 81: # Tl
        UI, UII, UIII = max(2.0, 0.63 + 3.35e-4 * T), 1.0, 2.0
    elif Z == 82: # Pb
        UI = max(1.0, 0.42 + 2.35e-4 * T)
        if T > 6125.0: UI = -1.2 + 5e-4 * T
        UII, UIII = max(2.0, 1.72 + 7.9e-5 * T), 1.0
    elif Z == 83: # Bi
        UI, UII, UIII = 2.78 + 2.87e-4 * T, max(1.0, 0.37 + 1.41e-4 * T), 2.5
    elif Z == 84: # Po
        UI, UII, UIII = 5.0, 5.0, 4.0
    elif Z == 85: # At
        UI, UII, UIII = 4.0, 6.0, 6.0
    elif Z == 86: # Rn
        UI, UII, UIII = 1.0, 4.0, 6.0
    elif Z == 87: # Fr
        UI, UII, UIII = 2.0, 1.0, 4.5
    elif Z == 88: # Ra
        UI, UII, UIII = 1.0, 2.0, 1.0
    elif Z == 89: # Ac
        UI, UII, UIII = 6.0, 3.0, 7.0
    elif Z == 90: # Th
        UI, UII, UIII = 8.0, 8.0, 8.0
    elif Z == 91: # Pa
        UI, UII, UIII = 50.0, 50.0, 50.0
    elif Z == 92: # U
        UI, UII, UIII = 25.0, 25.0, 25.0
        
    return UI, UII, UIII

# =====================================================================
# 92-Element Background Equation of State
# =====================================================================
def compute_background_eos(temp_array: np.ndarray, pg_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the Chemical Equilibrium for 92 elements in LTE to find the true 
    background electron density (Ne) and Hydrogen density (Nh).
    Adapted from cont_opacity.f90
    """
    
    DENO_SA = np.sum(10.0**(ABUND - 12.0))
    CSAHA1 = (h_CGS**2 / (2.0 * np.pi * m_e_CGS * kB_CGS))**1.5
    CSAHA2 = eV_CGS / kB_CGS
    
    ne_out = np.zeros_like(temp_array)
    nh_out = np.zeros_like(temp_array)
    
    for k in range(len(temp_array)):
        T = temp_array[k]
        Pg = pg_array[k]
        
        # Initial guess
        ne_new = Pg / (101.0 * kB_CGS * T)

        alpha1 = np.zeros(NELEM)
        alpha2 = np.zeros(NELEM)
        lamelec = CSAHA1 * (T**(-1.5))
        
        # Precompute Saha factors for all 92 elements at this depth
        for i in range(NELEM):
            UI, UII, UIII = get_partition_functions(i + 1, T)
            expo1 = np.clip(CSAHA2 * XI[i] / T, -650.0, 650.0)
            expo2 = np.clip(CSAHA2 * XII[i] / T, -650.0, 650.0)
            alpha1[i] = (UI / (2.0 * UII)) * lamelec * np.exp(expo1)
            alpha2[i] = (UII / (2.0 * UIII)) * lamelec * np.exp(expo2)

        error = 1.0
        iter_count = 1
        
        # Iterate until Ne converges
        while error > 1e-4 and iter_count <= 200:
            ne_old = ne_new
            n_minus_e = Pg - ne_new * kB_CGS * T    # Total particle density minus electrons
            nh = n_minus_e / (DENO_SA * kB_CGS * T) # Hydrogen number density
            nt = nh * 10.0**(ABUND - 12.0)          # Total number density of each element based on abundance and hydrogen density
            
            # Ionization fractions
            n_I = nt / (1.0 + (1.0 / (ne_new * alpha1)) * (1.0 + (1.0 / (ne_new * alpha2))))
            n_II = nt / (ne_new * alpha1 + 1.0 + (1.0 / (ne_new * alpha2)))
            n_III = nt / (1.0 + ne_new * alpha2 * (ne_new * alpha1 + 1.0))
            
            # Sum electrons (H- is index 0)
            ne_est = n_III[0] - n_I[0]
            ne_est += np.sum(n_II[1:]) + np.sum(n_III[1:])
            
            ne_new = (max(ne_est, 0.0) + ne_old) / 2.0
            error = abs((ne_new - ne_old) / ne_new) if ne_new > 0 else 0.0
            iter_count += 1
        
        if iter_count > 200:
            print(f"WARNING: Background EOS did not converge at iz={k}. Final error: {error:.2e}")

        ne_out[k] = ne_new
        nh_out[k] = nh
        
    return ne_out, nh_out


# =====================================================================
# Indices (0-based, Z-1) in the element tables for opacity species
# =====================================================================
_Z_IDX = {  # element symbol -> 0-based index in XI/XII/ABUND/MATOM
    'H':  0, 'He': 1, 'C':  5, 'N':  6, 'O':  7,
    'Mg': 11, 'Al': 12, 'Si': 13, 'Ca': 19, 'Fe': 25,
}

# The Wittmann saha() constant:  log10[ (2 pi me k / h^2)^1.5 ]  in CGS with Pe [dyne cm^-2]
_SAHA_LOG10_FAC = 9.0804625434325867


def _saha_ratio_neutral_to_ion(theta: float, eion_eV: float,
                                U_neutral: float, U_ion: float,
                                Pe: float) -> float:
    """
    Returns n(neutral)/n(ion) via the Wittmann saha() formula.
    Exactly mirrors wittmann.saha() but inverted so that large return values
    mean mostly neutral gas.
    """
    ratio_ion_to_neutral = (U_ion * 10.0**(_SAHA_LOG10_FAC - theta * eion_eV)
                            / (U_neutral * Pe * theta**2.5))
    return 1.0 / max(ratio_ion_to_neutral, 1.0e-300)


def _molecb(theta: float):
    """
    Wittmann polynomial fits for molecular equilibrium constants.
    Returns log10(P_H2 / P_H^2)  and  log10(P_H2+ / [P_H * P_H+]).
    Mirrors wittmann.molecb().
    """
    Y_H2  = -11.206998 + theta * (2.7942767 + theta * (7.9196803e-2 - theta * 2.4790744e-2))
    Y_H2p = -12.533505 + theta * (4.9251644 + theta * (-5.6191273e-2 + theta * 3.2687661e-3))
    return Y_H2, Y_H2p


def compute_background_species(temp_array: np.ndarray,
                                pg_array: np.ndarray,
                                ne_array: np.ndarray,
                                nh_array: np.ndarray) -> dict:
    """
    Given converged ne and nh from compute_background_eos, compute the
    number densities of every species required by the Wittmann continuous
    background-opacity functions.

    All densities labelled ``_per_U`` are divided by their partition function
    U(T), exactly as Wittmann's get_background_partials(divide_by_u=True).

    Returns
    -------
    dict with keys (each value is a 1-D array of shape (Ndepth,)):
        n_HI_per_U  : n(H I) / U(H I)          [cm^-3]
        n_HII       : n(H+)                      [cm^-3]
        n_Hminus    : n(H-)                      [cm^-3]
        n_H2        : n(H2)                      [cm^-3]
        n_HeI_per_U : n(He I) / U(He I)         [cm^-3]
        n_HeII_per_U: n(He+) / U(He+)           [cm^-3]
        n_HeIII     : n(He++)                    [cm^-3]
        n_CI_per_U  : n(C I) / U(C I)           [cm^-3]
        n_NI_per_U  : n(N I) / U(N I)           [cm^-3]
        n_OI_per_U  : n(O I) / U(O I)           [cm^-3]
        n_MgI_per_U : n(Mg I) / U(Mg I)         [cm^-3]
        n_MgII_per_U: n(Mg+) / U(Mg+)           [cm^-3]
        n_AlI_per_U : n(Al I) / U(Al I)         [cm^-3]
        n_SiI_per_U : n(Si I) / U(Si I)         [cm^-3]
        n_SiII_per_U: n(Si+) / U(Si+)           [cm^-3]
        n_CaII_per_U: n(Ca+) / U(Ca+)           [cm^-3]
        n_FeI_per_U : n(Fe I) / U(Fe I)         [cm^-3]
    """
    Ndepth = len(temp_array)
    CSAHA1 = (h_CGS**2 / (2.0 * np.pi * m_e_CGS * kB_CGS))**1.5
    CSAHA2 = eV_CGS / kB_CGS           # converts eV -> K

    keys = ['n_HI_per_U', 'n_HII', 'n_Hminus', 'n_H2',
            'n_HeI_per_U', 'n_HeII_per_U', 'n_HeIII',
            'n_CI_per_U', 'n_NI_per_U', 'n_OI_per_U',
            'n_MgI_per_U', 'n_MgII_per_U', 'n_AlI_per_U',
            'n_SiI_per_U', 'n_SiII_per_U', 'n_CaII_per_U',
            'n_FeI_per_U']
    result = {k: np.zeros(Ndepth) for k in keys}

    for k in range(Ndepth):
        T    = temp_array[k]
        ne   = max(ne_array[k], 1.0e-30)
        nh   = nh_array[k]
        Pe   = ne * kB_CGS * T
        theta = 5040.0 / T
        lamelec = CSAHA1 * T**(-1.5)

        # ------------------------------------------------------------------
        # HYDROGEN  (H-, H I, H II)
        # ------------------------------------------------------------------
        # Wittmann uses U(H I) = 2.0 (ground-state statistical weight g=2).
        # chemeq.get_partition_functions(1, T) returns UI=1.0 for the continuum
        # excited-state correction; we must use 2.0 to match wittmann.gasc().
        UI_H  = 2.0   # Wittmann convention
        UII_H = 1.0   # U(H+) = 1

        # Saha: H I -> H+ + e-  (ionisation potential = XII[0] = 13.595 eV)
        alpha1_H = (UI_H / (2.0 * UII_H)) * lamelec * np.exp(
            np.clip(CSAHA2 * XII[0] / T, -650.0, 650.0))
        frac_ion_H = 1.0 / max(ne * alpha1_H, 1.0e-300)  # n(H+)/n(H I)
        n_HI  = nh / (1.0 + frac_ion_H)
        n_HII = nh * frac_ion_H / (1.0 + frac_ion_H)

        result['n_HI_per_U'][k] = n_HI / UI_H   # = n_HI / 2
        result['n_HII'][k]      = n_HII

        # Saha: H I + e- -> H-  (electron affinity = XI[0] = 0.754 eV)
        # ratio p(H)/p(H-) mirrors wittmann.saha(theta, 0.754, U_H-=1, U_H=2, Pe)
        ratio_H_to_Hminus = (UI_H * 10.0**(_SAHA_LOG10_FAC - theta * XI[0])
                             / (1.0 * Pe * theta**2.5))
        result['n_Hminus'][k] = n_HI / max(ratio_H_to_Hminus, 1.0e-300)

        # Molecular hydrogen H2 via Wittmann's molecb polynomial
        # log10(P_H2 / P_H^2) = Y_H2
        Y_H2, _ = _molecb(theta)
        P_HI = n_HI * kB_CGS * T                      # partial pressure of H I
        P_H2 = P_HI**2 * 10.0**np.clip(Y_H2, -300.0, 300.0)
        result['n_H2'][k] = P_H2 / (kB_CGS * T)      # ideal gas

        # ------------------------------------------------------------------
        # HELIUM  (He I, He+, He++)
        # ------------------------------------------------------------------
        UI_He, UII_He, UIII_He = get_partition_functions(2, T)
        i_He = _Z_IDX['He']
        # alpha: n_e * alpha = n_neutral / n_ion
        a1_He = (UI_He  / (2.0 * UII_He )) * lamelec * np.exp(
                 np.clip(CSAHA2 * XI[i_He]  / T, -650.0, 650.0))
        a2_He = (UII_He / (2.0 * UIII_He)) * lamelec * np.exp(
                 np.clip(CSAHA2 * XII[i_He] / T, -650.0, 650.0))

        n_He_total = nh * 10.0**(ABUND[i_He] - 12.0)
        ne_a1 = ne * a1_He
        ne_a2 = ne * a2_He
        f_I_He   = n_He_total / (1.0 + 1.0/ne_a1 * (1.0 + 1.0/ne_a2))
        f_II_He  = n_He_total / (ne_a1 + 1.0 + 1.0/ne_a2)
        f_III_He = n_He_total / (1.0 + ne_a2 * (ne_a1 + 1.0))

        result['n_HeI_per_U'][k]  = f_I_He   / max(UI_He,   1.0)
        result['n_HeII_per_U'][k] = f_II_He  / max(UII_He,  1.0)
        result['n_HeIII'][k]      = f_III_He

        # ------------------------------------------------------------------
        # METAL NEUTRALS  (C, N, O, Mg, Al, Si, Fe)
        #   returns n(neutral)/U and n(singly-ionised)/U
        # ------------------------------------------------------------------
        def _metal(sym, key_I, key_II=None):
            idx = _Z_IDX[sym]
            UI_m, UII_m, UIII_m = get_partition_functions(idx + 1, T)
            a1_m = (UI_m  / (2.0 * UII_m )) * lamelec * np.exp(
                    np.clip(CSAHA2 * XI[idx]  / T, -650.0, 650.0))
            a2_m = (UII_m / (2.0 * UIII_m)) * lamelec * np.exp(
                    np.clip(CSAHA2 * XII[idx] / T, -650.0, 650.0))
            nt_m = nh * 10.0**(ABUND[idx] - 12.0)
            ne_a1_m = ne * a1_m
            ne_a2_m = ne * a2_m
            nI_m  = nt_m / (1.0 + 1.0/ne_a1_m * (1.0 + 1.0/ne_a2_m))
            result[key_I][k] = nI_m / max(UI_m, 1.0)
            if key_II is not None:
                nII_m = nt_m / (ne_a1_m + 1.0 + 1.0/ne_a2_m)
                result[key_II][k] = nII_m / max(UII_m, 1.0)

        _metal('C',  'n_CI_per_U')
        _metal('N',  'n_NI_per_U')
        _metal('O',  'n_OI_per_U')
        _metal('Mg', 'n_MgI_per_U', 'n_MgII_per_U')
        _metal('Al', 'n_AlI_per_U')
        _metal('Si', 'n_SiI_per_U', 'n_SiII_per_U')
        _metal('Fe', 'n_FeI_per_U')

        # Ca: only the singly-ionised stage (Ca II) is needed by LUKEOP
        idx_Ca = _Z_IDX['Ca']
        UI_Ca, UII_Ca, UIII_Ca = get_partition_functions(idx_Ca + 1, T)
        a1_Ca = (UI_Ca  / (2.0 * UII_Ca )) * lamelec * np.exp(
                 np.clip(CSAHA2 * XI[idx_Ca]  / T, -650.0, 650.0))
        a2_Ca = (UII_Ca / (2.0 * UIII_Ca)) * lamelec * np.exp(
                 np.clip(CSAHA2 * XII[idx_Ca] / T, -650.0, 650.0))
        nt_Ca = nh * 10.0**(ABUND[idx_Ca] - 12.0)
        ne_a1_Ca = ne * a1_Ca
        ne_a2_Ca = ne * a2_Ca
        nII_Ca = nt_Ca / (ne_a1_Ca + 1.0 + 1.0/ne_a2_Ca)
        result['n_CaII_per_U'][k] = nII_Ca / max(UII_Ca, 1.0)

    return result


# def partition_function(atom: MultiLevelAtom, stage: int, T: float) -> float:
#     """
#     Computes the Irwin partition function for a specific ionization stage at temperature T.
#     The polynomial follows the form ln Q = sum(a_i * (ln T)^i).
#     Calculated rapidly using nested multiplication (Horner's method) as recommended.
#     """
#     irwin_coeffs = atom.irwin_coefficients.get(stage, [])
    
#     if not irwin_coeffs:
#         return 1.0  # Fallback if no coefficients are provided

#     lnT = np.log(T)
#     sum_val = 0.0
    
#     # Rapid calculation using nested multiplication (Horner's method)
#     for a in reversed(irwin_coeffs):
#         sum_val = sum_val * lnT + a
        
#     return np.exp(sum_val)
