# psi(x) = sqrt(2/L)sin(kx)
# k = n * pi / L

import numpy
from quantum_functions import *
import matplotlib.pyplot as plt

pi = numpy.pi
sin = numpy.sin
sqrt = numpy.sqrt

L = x_max
n = 1
k = n * pi / L

# psi = sqrt(2 / L) * sin(k * x)
psi = sin(k * x)
norm_psi = normalise_psi(psi)

expected_E = n ** 2 * pi ** 2 * h_bar ** 2 / (2 * m * L ** 2)
E = energy_expectation(psi)
print("Exp:", expected_E)
H = hamiltonian(potential(x))
print("E:", E)

plt.plot(x, psi)
plt.xlabel("x")
plt.ylabel("$\psi$")
plt.show()

plt.plot(x, normalise_psi(psi))
plt.xlabel("x")
plt.ylabel("Norm $\psi$")
plt.show()
