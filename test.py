import numpy as np
import matplotlib.pyplot as plt

# Define the Hamiltonian operator
def hamiltonian(x):
    return -0.5 * np.gradient(np.gradient(x)) + 0.5 * x ** 2

# Solve the Schrödinger equation for the harmonic oscillator potential
x = np.linspace(-5, 5, 1000)
x_2d = x[:, np.newaxis]  # reshape x to a 2D array
eigenvalues, eigenfunctions = np.linalg.eigh(hamiltonian(x_2d))

# Calculate the energy levels corresponding to the eigenvalues
energy_levels = eigenvalues + 0.5

# Plot the energy levels
plt.plot(x, energy_levels, 'k-')

# Plot the eigenfunctions for the first few energy levels
for i in range(5):
    plt.plot(x, eigenfunctions[:, i] + energy_levels[i], 'r-')

plt.xlabel('Position')
plt.ylabel('Energy')
plt.show()

