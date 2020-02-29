import random
import time

import matplotlib.pyplot as plt
import numpy
import scipy.linalg as linalg

# global constants:
hbar = 6.582119569 * 10 ** -16  # 6.582119569x10^16 (from wikipedia)
m = 9.1093837015 * 10 ** -31  # 9.1093837015(28)x10^31
factor = -(hbar ** 2) / (2 * m)


def normalise(psi: numpy.ndarray, dx: float):
    # integrate using the rectangular rule
    norm = numpy.sum(psi * psi) * dx
    norm_psi = psi / numpy.sqrt(norm)
    return norm_psi


global A


def generate_derivative_matrix(dimensions: int, dx):
    global A
    A = numpy.zeros((dimensions, dimensions))
    for i in range(1, dimensions - 1):
        A[i][i - 1], A[i][i], A[i][i + 1] = 1, -2, 1
    A[0, 0], A[0, 1], A[0][2], A[-1, -1], A[-1, -2], A[-1, -3] = 1, -2, 1, 1, -2, 1
    return A * (dx ** -2)


def energy(psi: numpy.ndarray, V: numpy.ndarray, dx: float):
    Vp = V * psi
    Tp = factor * (A @ psi)
    return numpy.nansum(psi * (Tp + Vp)) * dx


def potential(x: numpy.ndarray):
    length = len(x)
    third = length // 3
    # mid, bef = numpy.zeros(third + 1), numpy.linspace(numpy.inf, numpy.inf, third)
    mid, bef = numpy.zeros(third + 1), numpy.linspace(10, 10, third)
    aft = bef.copy()
    return numpy.concatenate((bef, mid, aft))

    # return 0.5 * x ** 2


def gen_orthonormal_states(pre_existing_states: numpy.ndarray, size):
    # there are no known states already
    if pre_existing_states.size == 0:
        return numpy.identity(size)
    else:
        orthonormal_states = linalg.null_space(pre_existing_states)
        n = len(pre_existing_states)
        for j in range(n):
            for k in range(len(orthonormal_states[n])):
                orthonormal_states[j][k] = 0

        return orthonormal_states.transpose()


def nth_state(start: float, stop: float, dimension: int, num_iterations: int, previous_states: numpy.ndarray):
    # the iteration number
    n = 0
    if previous_states.size != 0:
        n = previous_states.shape[0]

    t1 = time.time()
    states = gen_orthonormal_states(previous_states, dimension)
    row_size = states.shape[0]

    random.seed("THE-VARIATIONAL-PRINCIPLE")

    dx = (stop - start) / dimension

    x = numpy.linspace(start, stop, dimension)
    V = potential(x)

    psi = numpy.ones(dimension)
    psi[0], psi[-1] = 0, 0

    # handling for the inf values in the infinite square well, or similar:
    for j in range(len(psi)):
        if numpy.isnan(V[j]) or numpy.isinf(V[j]):
            psi[j] = 0

    psi = normalise(psi, dx)

    previous_energy = energy(psi, V, dx)
    print("Initial Energy:", previous_energy)

    for i in range(num_iterations):
        rand_x = random.randrange(1, row_size - 1)

        # handling for inf values from V:
        if numpy.isnan(V[rand_x]) or numpy.isinf(V[rand_x]):
            continue

        rand_y = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_y *= -1

        psi += states[rand_x] * rand_y
        psi = normalise(psi, dx)

        new_energy = energy(psi, V, dx)
        if new_energy < previous_energy:
            previous_energy = new_energy
        else:
            psi -= states[rand_x] * rand_y
            psi = normalise(psi, dx)

    print("Final Energy:", energy(psi, V, dx))
    t2 = time.time()
    print("The time for the " + str(n) + "th iteration is:", t2 - t1, "s.\n")

    # Correction of artifacts at edge:
    for j in range(n + 1):
        psi[j] = 0
    psi = normalise(psi, dx)

    plt.plot(x, psi)
    plt.title("The {}th State for the Finite Square Well:".format(n))
    plt.ylabel("$\psi$")
    plt.xlabel("x")
    plt.show()

    return psi


def main():
    a, b, N, num_iterations = -10, 10, 100, 10 ** 5
    x = numpy.linspace(a, b, N)

    dx = (b - a) / N
    generate_derivative_matrix(N, dx)
    existing_states = numpy.array([])
    number_states = 5
    for i in range(number_states):
        psi = nth_state(a, b, N, num_iterations, existing_states)
        if existing_states.size == 0:
            existing_states = numpy.array([psi])
        else:
            existing_states = numpy.vstack((existing_states, psi))

    for j in range(existing_states.shape[0]):
        plt.plot(x, existing_states[j])

    plt.title("Wavefunctions $\psi$ for the Finite Square Well:")
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    # # plt.legend(("Original $\psi$", "potential", "Normalised $\psi$", "Final $\psi$"))
    # plt.legend(("Potential", "Ground State", "Second State", "Third State", "Fourth State", "..."))
    plt.legend(("Ground State", "Second State", "Third State", "Fourth State", "..."))
    # # plt.legend(("Ground State", "Analytical Solution"))
    plt.show()

    ground_psi = existing_states[0]
    orthonormal_states = gen_orthonormal_states(existing_states, N)
    for j in range(len(orthonormal_states)):
        if abs(orthonormal_states[j][j]) > 0.01:
            plt.plot(x, orthonormal_states[j])
    plt.title("Error in Orthonormal States:")
    plt.show()


main()
