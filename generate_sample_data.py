#!/usr/bin/env python3
"""
Generate sample master curve data for testing the fitting functionality
"""

import numpy as np

# Create sample data using known parameters
E0 = 15.0
E_i = [800.0, 4000.0, 12000.0, 18000.0]
tau_i = [1e-6, 1e-4, 1e-2, 1.0]

# Generate frequency range
freq_log = np.linspace(-8, 15, 100)  # log10(Hz)
freq = 10**freq_log  # Hz
omega = 2 * np.pi * freq  # rad/s

# Calculate E' and E"
E_prime = np.zeros_like(omega)
E_double_prime = np.zeros_like(omega)

E_prime += E0
for E, tau in zip(E_i, tau_i):
    E_prime += E * (omega * tau)**2 / (1 + (omega * tau)**2)
    E_double_prime += E * omega * tau / (1 + (omega * tau)**2)

# Add some noise to make it realistic
np.random.seed(42)
noise_level = 0.03
E_prime *= (1 + noise_level * np.random.randn(len(E_prime)))
E_double_prime *= (1 + noise_level * np.random.randn(len(E_double_prime)))

# Save to CSV file (log10(f), log10(E'), log10(E"))
data = np.column_stack([freq_log, np.log10(E_prime), np.log10(E_double_prime)])

header = "log10_frequency_Hz,log10_storage_modulus_MPa,log10_loss_modulus_MPa"
np.savetxt('sample_master_curve.csv', data, delimiter=',', header=header, comments='')

print("Sample master curve data generated successfully!")
print(f"File: sample_master_curve.csv")
print(f"Data points: {len(freq_log)}")
print(f"\nTrue parameters used:")
print(f"E0 = {E0} MPa")
print(f"E_i = {E_i}")
print(f"tau_i = {tau_i}")
print(f"\nNoise level: {noise_level*100}%")
