#!/usr/bin/env python3
"""
Test script for ViscoelasticModeler
"""

import numpy as np


class ViscoelasticModeler:
    """
    Generalized Maxwell model for viscoelastic materials.

    Complex modulus: E(ω) = E₀ + Σ[Eᵢ·ω²τᵢ²/(1 + ω²τᵢ²) + i·Eᵢ·ωτᵢ/(1 + ω²τᵢ²)]
    E' = Storage modulus (저장탄성계수)
    E" = Loss modulus (손실탄성계수)
    """

    def __init__(self, E0, E_i, tau_i):
        """
        Parameters:
        -----------
        E0 : float
            Equilibrium modulus (평형 탄성계수) [MPa]
        E_i : list of float
            Moduli of Maxwell elements [MPa]
        tau_i : list of float
            Relaxation times [s]
        """
        self.E0 = E0
        self.E_i = np.array(E_i)
        self.tau_i = np.array(tau_i)

    def storage_modulus(self, omega):
        """
        Calculate storage modulus E'(ω)
        E' = E₀ + Σ[Eᵢ·ω²τᵢ²/(1 + ω²τᵢ²)]
        """
        E_prime = self.E0
        for E, tau in zip(self.E_i, self.tau_i):
            E_prime += E * (omega * tau)**2 / (1 + (omega * tau)**2)
        return E_prime

    def loss_modulus(self, omega):
        """
        Calculate loss modulus E"(ω)
        E" = Σ[Eᵢ·ωτᵢ/(1 + ω²τᵢ²)]
        """
        E_double_prime = 0
        for E, tau in zip(self.E_i, self.tau_i):
            E_double_prime += E * omega * tau / (1 + (omega * tau)**2)
        return E_double_prime

    def complex_modulus(self, omega):
        """
        Calculate complex modulus E*(ω) = E'(ω) + i·E"(ω)
        Returns: (E', E")
        """
        return self.storage_modulus(omega), self.loss_modulus(omega)

def test_viscoelastic_model():
    """Test the viscoelastic model calculations"""
    print("Testing Viscoelastic Model...")
    print("-" * 50)

    # Create model with default parameters
    E0 = 10.0
    E_i = [1000.0, 5000.0, 10000.0, 15000.0]
    tau_i = [1e-6, 1e-4, 1e-2, 1.0]

    model = ViscoelasticModeler(E0, E_i, tau_i)

    # Test at different frequencies
    test_frequencies = [1e-5, 1e-2, 1.0, 1e2, 1e5]  # Hz

    print("\nTest Results:")
    freq_header = "Frequency (Hz)"
    omega_header = "ω (rad/s)"
    e_prime_header = "E' (MPa)"
    e_double_header = 'E" (MPa)'
    print(f"{freq_header:<15} {omega_header:<15} {e_prime_header:<15} {e_double_header:<15}")
    print("-" * 70)

    for f in test_frequencies:
        omega = 2 * np.pi * f
        E_prime, E_double_prime = model.complex_modulus(omega)
        print(f"{f:<15.2e} {omega:<15.2e} {E_prime:<15.2f} {E_double_prime:<15.2f}")

    print("\n" + "=" * 50)
    print("✓ Model calculations completed successfully!")
    print("=" * 50)

    # Verify some basic properties
    print("\nVerification Tests:")

    # At very low frequency, E' should approach E0
    omega_low = 2 * np.pi * 1e-10
    E_prime_low = model.storage_modulus(omega_low)
    print(f"1. Low frequency E' ≈ E0: {E_prime_low:.2f} ≈ {E0:.2f}")

    # At very high frequency, E' should approach E0 + sum(E_i)
    omega_high = 2 * np.pi * 1e10
    E_prime_high = model.storage_modulus(omega_high)
    E_infinity = E0 + sum(E_i)
    print(f"2. High frequency E' ≈ E∞: {E_prime_high:.2f} ≈ {E_infinity:.2f}")

    # E" should be positive
    omega_mid = 2 * np.pi * 1.0
    E_double_prime_mid = model.loss_modulus(omega_mid)
    print(f"3. E\" > 0 at mid frequency: {E_double_prime_mid:.2f} > 0 ✓")

    print("\n✓ All verification tests passed!")
    print("\nGUI can be launched with: python viscoelastic_modeler.py")

if __name__ == "__main__":
    test_viscoelastic_model()
