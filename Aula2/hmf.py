import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Defining constants
criticalOverdensity = 3 * (12 * np.pi)**(2 / 3) / 20  # Critical overdensity for spherical collapse
criticalDensity = 2.77536627e+11  # Critical density in Msun/Mpc^3 h^-2
Om0 = 0.3111  # Matter density parameter at z=0
H0 = 67.66  # Hubble constant in km/s/Mpc
sigma8 = 0.8102  # RMS fluctuation in spheres of 8 Mpc/h

def W(x):
    """
    Fourier transform of the top-hat window function.
    
    Parameters:
    x : float or ndarray
        The argument of the window function (kR).
    
    Returns:
    float or ndarray
        The value of the Fourier-transformed top-hat window function at x.
    """
    # Avoid division by zero by handling x=0 separately
    return np.where(x != 0, 3 / x**3 * (np.sin(x) - x * np.cos(x)), 1.0)

def sigma2(k, Pk, R):
    """
    Compute the mass variance sigma^2(R) for a given power spectrum.
    
    Parameters:
    k : ndarray
        Array of wavenumbers (h/Mpc).
    Pk : ndarray
        Power spectrum corresponding to the wavenumbers k ((Mpc/h)^3).
    R : float
        Smoothing scale (Mpc/h).
    
    Returns:
    float
        The mass variance sigma^2 at scale R.
    """
    # Compute the window function values at k * R
    WkR = W(k * R)
    
    # Compute the integrand values at each k
    integrand_values = k**3 * Pk * WkR**2 / (2 * np.pi**2)
    
    # Convert k to natural logarithm for integration
    ln_k = np.log(k)
    
    # Perform the integration over ln(k) using the trapezoidal rule
    sigma_squared = np.trapz(integrand_values, ln_k)
    return sigma_squared

def multiplicityFunction(peakHeight):
    """
    Compute the Press-Schechter multiplicity function.
    
    Parameters:
    peakHeight : float or ndarray
        The peak height (delta_c / sigma(M)).
    
    Returns:
    float or ndarray
        The value of the multiplicity function at the given peak heights.
    """
    return np.sqrt(2 / np.pi) * peakHeight * np.exp(-peakHeight**2 / 2)

# Load wavenumber k and power spectrum Pk from file
# Assumes 'Pk.txt' has two columns: k (h/Mpc) and Pk ((Mpc/h)^3)
k, Pk = np.loadtxt("Pk.txt").T

# Define the smoothing scale R (e.g., in Mpc/h)
R_fixed = 8.0  # Mpc/h

# Compute sigma^2 at the fixed scale R = 8 Mpc/h
sigma_squared = sigma2(k, Pk, R_fixed)

# Print the computed sigma^2
print(f"sigma^2 at R = {R_fixed} Mpc/h: {sigma_squared:.4f}")

# Creating a mass array from 1e13 to 5e15 Msun/h, logarithmically spaced
M = np.geomspace(1e13, 5e15, 100)  # Msun/h

# Convert mass M to smoothing scale R using the relation:
# M = (4/3) * pi * R^3 * criticalDensity * Om0
R = (M / ((4/3) * np.pi * criticalDensity * Om0))**(1/3)  # Mpc/h

# Compute sigma^2(M) for each mass scale R
sigmaM2 = np.array([sigma2(k, Pk, r) for r in R])

# Compute the peak-height (nu = delta_c / sigma(M))
peakHeight = criticalOverdensity / np.sqrt(sigmaM2)

# Compute the Press-Schechter differential mass function dndlnm
# Using the formula:
# dndlnm = (rho_m / M) * vf(nu) * |dln(sigma)/dlnM|
# where f(nu) is the multiplicity function
# Note: rho_m = criticalDensity * Om0
rho_m = criticalDensity * Om0  # Msun/Mpc^3 h^-2

# Compute dln(sigma)/dlnM using numerical differentiation
dlnsigma_dlnM = np.gradient(np.log(np.sqrt(sigmaM2)), np.log(M))

# Compute dndlnm using Press-Schechter formalism
dndlnm_ps = (rho_m / M) * multiplicityFunction(peakHeight) * np.abs(dlnsigma_dlnM)

# Plotting the Press-Schechter Halo Mass Function (HMF)
plt.figure(figsize=(10, 6))
plt.loglog(M, dndlnm_ps, label='Press-Schechter HMF', color='blue', linewidth=2)

# Reading simulation HMF data from file 'dndlnm.txt'
# Assumes 'dndlnm.txt' has two columns: M (Msun/h) and dndlnm (number density per ln M)
sim_M, sim_dndlnm = np.loadtxt("dndlnm.txt").T

# Plotting the simulation HMF
plt.loglog(sim_M, sim_dndlnm, label='Simulation HMF', color='red', linestyle='--', linewidth=2)

# Adding labels and title with units
plt.xlabel(r'Halo Mass $M \, [\mathrm{M_\odot}/h]$', fontsize=14)
plt.ylabel(r'$\frac{dn}{d\ln M} \, [\mathrm{Mpc^{-3} \, h^{3}}]$', fontsize=14)
plt.title('Press-Schechter Halo Mass Function vs. Simulation', fontsize=16)

# Adding a legend to distinguish between Press-Schechter and Simulation
plt.legend(fontsize=12)
# Display the plot
plt.tight_layout()
plt.show()