import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# global constants:
hbar = 6.582119569 * 10 ** -16  # 6.582119569x10^-16 (from wikipedia)
m = 9.1093837015 * 10 ** -31  # 9.1093837015(28)x10^-31
factor = -(hbar ** 2) / (2 * m)


def normalise(psi: np.ndarray, dx: float):
    # integrate using the rectangular rule
    norm = np.sum(psi * psi) * dx
    norm_psi = psi / np.sqrt(norm)
    return norm_psi


global A


def generate_derivative_matrix(dimensions: int, dx):
    global A
    A = np.zeros((dimensions, dimensions))
    for i in range(1, dimensions - 1):
        A[i][i - 1], A[i][i], A[i][i + 1] = 1, -2, 1
    A[0][0], A[0][1], A[-1][-1], A[0][2], A[-1][-2], A[-1][-3] = 1, -2, 1, 1, -2, 1
    return A * (dx ** -2)


def energy(psi: np.ndarray, V: np.ndarray, dx: float):
    Vp = V * psi
    # filter out nan values in Vp
    Vp = np.where(np.isfinite(Vp), Vp, 0)
    # A is the 2nd derivative matrix.
    Tp = factor * (A @ psi)
    return np.sum(psi * (Tp + Vp)) * dx


def potential(x: np.ndarray):
    length = len(x)
    third = length // 3
    mid, bef = np.zeros(third + 1), np.linspace(np.inf, np.inf, third)
    # mid, bef = numpy.zeros(third + 1), numpy.linspace(10, 10, third)
    aft = bef.copy()
    return np.concatenate((bef, mid, aft))

    # return 0.5 * x ** 2


def gen_orthonormal_states(pre_existing_states: np.ndarray, size):
    # there are no known states already
    if pre_existing_states.size == 0:
        return np.identity(size)
    else:
        orthonormal_states = la.null_space(pre_existing_states)
        n = len(pre_existing_states)
        # artifacts fix
        for j in range(n):
            for k in range(len(orthonormal_states[n])):
                orthonormal_states[j][k] = 0

        return orthonormal_states.transpose()


def nth_state(start: float, stop: float, dimension: int, num_iterations: int, previous_states: np.ndarray):
    # the iteration number
    n = 0
    if previous_states.size != 0:
        n = previous_states.shape[0]

    t1 = time.time()
    orthonormal_states = gen_orthonormal_states(previous_states, dimension)
    row_size = orthonormal_states.shape[0]

    random.seed("THE-VARIATIONAL-PRINCIPLE")

    dx = (stop - start) / dimension

    x = np.linspace(start, stop, dimension)
    V = potential(x)

    psi = np.ones(dimension)
    psi[0], psi[-1] = 0, 0

    # handling for the inf values in the infinite square well, or similar:
    for j in range(len(psi)):
        if not np.isfinite(V[j]):
            psi[j] = 0

    plt.plot(orthonormal_states)
    plt.title("STATES BEFORE V CHECK: " + str(n))
    plt.show()

    # infinite fix
    for k in range(len(psi)):
        if not np.isfinite(V[k]):
            for j in range(len(orthonormal_states)):
                orthonormal_states[j, k] = 0
    # TODO ^^^ does orthonormal_states have to be re-normalised after change?

    plt.plot(orthonormal_states)
    plt.title("STATES AFTER V CHECK: " + str(n))
    plt.show()

    psi = normalise(psi, dx)

    prev_E = energy(psi, V, dx)
    print("Initial Energy:", prev_E)

    for i in range(num_iterations):
        rand_x = random.randrange(1, row_size - 1)

        rand_y = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_y *= -1

        psi += orthonormal_states[rand_x] * rand_y
        psi = normalise(psi, dx)

        new_E = energy(psi, V, dx)
        if new_E < prev_E:
            prev_E = new_E
        else:
            psi -= orthonormal_states[rand_x] * rand_y
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
    x = np.linspace(a, b, N)

    dx = (b - a) / N
    generate_derivative_matrix(N, dx)
    existing_states = np.array([np.zeros(N)])
    number_states = 5
    for i in range(number_states):
        # print(existing_states)
        for j in range(len(existing_states)):
            plt.plot(existing_states[j])
        # plt.plot(existing_states)
        plt.title("EXISTING STATES")
        plt.show()
        psi = nth_state(a, b, N, num_iterations, existing_states)

        if existing_states.size == 0:
            existing_states = np.array([psi])
        else:
            existing_states = np.vstack((existing_states, psi))

    for j in range(existing_states.shape[0]):
        plt.plot(x, existing_states[j])

    plt.title("Wavefunctions $\psi$ for the Finite Square Well:")
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    plt.legend(("Ground State", "Second State", "Third State", "Fourth State", "..."))
    plt.show()


main()
