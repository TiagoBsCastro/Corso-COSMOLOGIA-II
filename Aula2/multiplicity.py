import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Defining constants
criticalOverdensity = 3 * (12 * np.pi)**(2 / 3) / 20  # Critical overdensity for spherical collapse
criticalDensity = 2.77536627e+11  # Critical density in Msun/Mpc^3 h^-2
Om0 = 0.3111  # Matter density parameter at z=0
H0 = 67.66  # Hubble constant in km/s/Mpc
sigma8 = 0.8102  # RMS fluctuation in spheres of 8 Mpc/h

masses = np.loadtxt("halo_masses_vir_z0.0.txt")
