import random

import matplotlib.pyplot as plt
import numpy

# global constants:
hbar = 1
m = 1
factor = -(hbar ** 2) / (2 * m)


def normalise(psi, dx):
    # integrate using the rectangular rule
    norm = numpy.sum(psi * psi) * dx
    norm_psi = psi / numpy.sqrt(norm)
    return norm_psi


def energy(psi, V, dx):
    total_energy = 0
    for i in range(1, len(psi) - 1):
        # use central difference:
        T = factor * (psi[i - 1] - 2 * psi[i] + psi[i + 1]) * (dx ** -2)
        Vp = V[i] * psi[i]
        E = psi[i] * (T + Vp)

        total_energy += E * dx

    return total_energy


def potential(x):
    return 0.5 * x ** 2


def main():
    random.seed("REVAMP")

    a, b, N = -10, 10, 100
    dx = (b - a) / N
    x = numpy.linspace(a, b, N)

    psi = numpy.ones(N)
    # psi = numpy.exp(- x**2 * 0.5)
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
        rand_x = random.randrange(1, N - 1)
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
        # old_E, new_E = new_E, energy(psi, V, dx)
        #
        # if new_E > old_E:
        #     new_E = old_E
        #     psi[rand_x] -= rand_y
        #     psi = normalise(psi, dx)

    plt.plot(x, psi * scale)
    plt.plot(x, normalise(numpy.exp(-0.5 * x ** 2), dx) * scale)
    plt.legend(("Original $\psi$", "potential", "Normalised $\psi$", "Final $\psi$", "Actual $\psi$"))
    plt.show()
    print("Final Energy:", prev_E)


main()
