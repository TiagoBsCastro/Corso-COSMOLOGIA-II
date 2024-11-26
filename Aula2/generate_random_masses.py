import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
from colossus.cosmology import cosmology
from colossus.lss import mass_function

# Set up the cosmology
cosmology.setCosmology('planck18')

# Define the volume in (Mpc/h)^3
volume = (2000.0)**3  # 2 Gpc/h cube

# Mass definitions and redshifts
mass_defs = ['vir', '500c']
redshifts = [0.0, 1.0]

# Mass range (Msun/h)
M_min = 1e13
M_max = 1e16
n_mass = 1000  # Number of mass bins

# Loop over mass definitions and redshifts
for mass_def in mass_defs:
    for z in redshifts:
        print(f"Generating masses for mass definition: {mass_def}, redshift: {z}")

        # Create mass array
        M = np.geomspace(M_min, M_max, n_mass)

        # Calculate differential mass function (dn/dlogM) in units of h^3 Mpc^-3
        dndlogM = mass_function.massFunction(M, z, mdef=mass_def, model='tinker08', q_out='dndlnM')

        # Compute the cumulative number density
        tot = np.trapz(dndlogM, np.log(M))
        cumulative_number_density = tot - cumtrapz(dndlogM, np.log(M))
        cumulative_number_density = np.abs(np.insert(cumulative_number_density, 0, tot))

        # Total number of halos in the volume
        total_number_density = cumulative_number_density[0]  # Number density above M_min
        total_number = total_number_density * volume

        print(f"Total number of halos: {int(total_number):,}")

        # Generate random cumulative numbers
        n_halos = int(np.round(total_number))
        cumulative_numbers = np.random.uniform(0, total_number_density, n_halos)

        # Interpolate to find the masses corresponding to the random cumulative numbers
        mass_function_interp = np.interp(cumulative_numbers, cumulative_number_density[::-1], M[::-1])

        # Save the masses to a file
        filename = f"halo_masses_{mass_def}_z{z}.txt"
        np.savetxt(filename, mass_function_interp, header='Halo masses (Msun/h)')
        print(f"Saved halo masses to {filename}\n")

        # Optional: Plot the mass function and the histogram of generated masses
        plt.figure(figsize=(8, 6))
        plt.loglog(M, dndlogM, label='Tinker Mass Function')
        hist, bins = np.histogram(mass_function_interp, bins=np.geomspace(mass_function_interp.min(), mass_function_interp.max()))
        masses, bins = np.histogram(mass_function_interp, bins=bins, weights=mass_function_interp)
        masses[hist>0] = masses[hist>0]/hist[hist>0]
        masses[hist==0] = ((bins[1:] + bins[:-1])/2)[hist==0]
        plt.loglog(masses, hist/volume/np.log(bins[1:]/bins[:-1]), label='Generated Masses')
        plt.xlabel(r'Halo Mass $M\ [M_\odot/h]$')
        plt.ylabel(r'$dn/d\ln M\ [h^3\,\mathrm{Mpc}^{-3}\,h^3]$')
        plt.title(f'Mass Definition: {mass_def}, Redshift: {z}')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.show()
