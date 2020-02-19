import random

import matplotlib.pyplot as plt
import numpy

# global constants:
hbar = 1
m = 1
factor = -(hbar ** 2) / (2 * m)


def normalise(psi: numpy.ndarray, dx: float):
    # integrate using the rectangular rule
    norm = numpy.sum(psi * psi) * dx
    norm_psi = psi / numpy.sqrt(norm)
    return norm_psi


def second_derivative(f: numpy.ndarray, i: int, dx: float):
    inv_h_sq = dx ** -2
    # forward difference for left edge:
    if i == 0:
        integrand = f[i] - 2 * f[i + 1] + f[i + 2]
    # backward difference for right edge:
    elif i == len(f) - 1:
        integrand = f[i - 2] - 2 * f[i - 1] + f[i]
    # central difference for non edge cases
    else:
        integrand = f[i - 1] - 2 * f[i] + f[i + 1]

    return integrand * inv_h_sq


def energy(psi: numpy.ndarray, V: numpy.ndarray, dx: float):
    total_energy = 0
    # for i in range(1, len(psi) - 1):
    for i in range(len(psi)):
        T = factor * second_derivative(psi, i, dx)
        Vp = V[i] * psi[i]
        E = psi[i] * (T + Vp)

        total_energy += E

    return total_energy * dx


def potential(x: numpy.ndarray):
    return 0.5 * x ** 2


def main():
    random.seed("THE-VARIATIONAL-PRINCIPLE")

    a, b, N = -10, 10, 100
    dx = (b - a) / N
    x = numpy.linspace(a, b, N)

    psi = numpy.ones(N)
    actual_psi = normalise(numpy.exp(- x ** 2 * 0.5), dx)
    psi[0], psi[-1] = 0, 0
    V = potential(x)

    scale = 100
    plt.plot(x, psi)
    psi = normalise(psi, dx)

    plt.plot(x, V)
    plt.plot(x, psi * scale)
    # plt.show()

    # new_E = energy(psi, V, dx)
    prev_E = energy(psi, V, dx)
    print("initial energy:", prev_E)

    num_iterations = 100000
    for i in range(num_iterations):
        # rand_x = random.randrange(1, N - 1)
        rand_x = random.randrange(N)
        rand_y = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_y *= -1

        psi[rand_x] += rand_y
        psi = normalise(psi, dx)

        new_E = energy(psi, V, dx)
        if new_E < prev_E:
            prev_E = new_E
        else:
            psi[rand_x] -= rand_y
            psi = normalise(psi, dx)

    plt.plot(x, psi * scale)
    plt.plot(x, actual_psi * scale)
    plt.title("$\psi$ functions")
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    plt.legend(("Original $\psi$", "potential", "Normalised $\psi$", "Final $\psi$", "Actual $\psi$"))
    plt.show()
    print("Final Energy:", prev_E)

    diff_psi = actual_psi - psi
    plt.plot(x, diff_psi)
    plt.title("Difference in $\psi$ calculation:")
    plt.xlabel("x")
    plt.ylabel("difference in $\psi$")
    plt.show()
    print("Integral of Difference:", numpy.sum(diff_psi) * dx)


main()
