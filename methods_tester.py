import numpy

a = -20
b = 20
num_samples = 2 ** 5 + 1
step_size = (b - a) / (num_samples - 1)

# # Newton-Cotes formula:
import scipy.integrate as integrate

A, error_coefficient = integrate.newton_cotes(num_samples - 1, 1)


def NT_integrate(f: numpy.ndarray, i: int, step=step_size):
    return (A[i] * f[i]) * step


f = numpy.linspace(a, b, num_samples)
F = numpy.zeros(num_samples)
for i in range(num_samples):
    F[i] = NT_integrate(f, i)

import matplotlib.pyplot as plt

plt.plot(A)
plt.title("A")
plt.show()

plt.plot(f)
plt.title("f")
plt.show()

plt.plot(F)
plt.title("F")
plt.show()

print("Newton-Cotes sum:", numpy.sum(F))
print("Trapeziodal Rule:", integrate.trapz(f))
print("Simpsons Rule:", integrate.simps(f))
print("Romberg:", integrate.romb(f))
