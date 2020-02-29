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


def generate_derivative_matrix(axis_length: int, dr: float):
    global A
    A = np.zeros((axis_length, axis_length))
    for i in range(1, axis_length - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = 1, -2, 1
    A[0, 0], A[0, 1], A[0, 2], A[-1, -1], A[-1, -2], A[-1, -3] = 1, -2, 1, 1, -2, 1
    return A * (dr ** -2)


def energy(psi: np.ndarray, V: np.ndarray, dr: float):
    # when V is inf, wil get an invalid value error at runtime, not an issue, is sorted in filtering below:
    Vp = V * psi
    # filter out nan values in Vp
    Vp = np.where(np.isfinite(Vp), Vp, 0)
    # A is the 2nd derivative matrix.
    Tp = np.empty(psi.shape)
    for ax in range(len(psi)):
        Tp[ax] = A @ psi[ax]
    Tp *= factor

    E = 1
    for ax in range(len(psi)):
        E *= np.sum(psi[ax] * (Tp[ax] + Vp[ax]))
    return E * dr


def potential(r: np.ndarray):
    # length = len(x)
    # third = length // 3
    # mid, bef = np.zeros(third + 1), np.linspace(np.inf, np.inf, third)
    # # mid, bef = np.zeros(third + 1), np.linspace(10, 10, third)
    # aft = bef.copy()
    # return np.concatenate((bef, mid, aft))

    return 0.5 * r ** 2


def gen_orthonormal_states(pre_existing_states: np.ndarray, num_axes, axis_size, fix_artifacts=True):
    # there are no known states already
    if pre_existing_states.size == 0:
        I = np.empty((num_axes, axis_size, axis_size))
        for ax in range(num_axes):
            I[ax] = np.identity(axis_size)
        return I
        # return np.identity(axis_size)
    else:
        orthonormal_states = la.null_space(pre_existing_states)
        n = len(pre_existing_states)

        # artifacts fix
        if fix_artifacts:
            for j in range(n):
                for k in range(len(orthonormal_states[n])):
                    orthonormal_states[j, k] = 0

        return orthonormal_states.T


def nth_state(start: float, stop: float, num_axes: int, axis_length: int, num_iterations: int,
              previous_states: np.ndarray,
              fix_infinities=True, fix_artifacts=True):
    n = (len(previous_states) // num_axes) - 1

    t1 = time.time()
    # TODO error in inf occurs because null_space returned is wrong?
    #  occurs because 1st state == 0th state => orthonormals goosed.
    #  therefore: make 1 good -> all good?

    orthonormal_states = gen_orthonormal_states(previous_states, num_axes, axis_length)
    row_size = len(orthonormal_states)

    random.seed("THE-VARIATIONAL-PRINCIPLE")

    dr = (stop - start) / axis_length

    x = np.linspace(start, stop, axis_length)
    r = np.empty((num_axes, axis_length))
    for ax in range(num_axes):
        r[ax] = x.copy()

    V = potential(r)

    psi = np.ones((num_axes, axis_length))
    for ax in range(num_axes):
        psi[ax, 0], psi[ax, -1] = 0, 0

    if fix_infinities:
        for ax in range(num_axes):
            # handling for the inf values in the infinite square well, or similar:
            for j in range(len(psi[ax])):
                if not np.isfinite(V[ax, j]):
                    psi[ax, j] = 0

            # infinite fix
            for k in range(len(psi[ax])):
                if not np.isfinite(V[ax, k]):
                    for j in range(len(orthonormal_states)):
                        orthonormal_states[ax, j, k] = 0
            # TODO ^^^ does orthonormal_states have to be re-normalised after change? ... no?

    psi = normalise(psi, dr)

    prev_E = energy(psi, V, dr)
    print("Initial Energy:", prev_E)

    for i in range(num_iterations):
        rand_index = random.randrange(1, row_size - 1)

        rand_change = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_change *= -1

        orthonormal_basis = orthonormal_states[rand_index]

        psi += orthonormal_basis * rand_change
        psi = normalise(psi, dr)

        new_E = energy(psi, V, dr)
        if new_E < prev_E:
            prev_E = new_E
        else:
            psi -= orthonormal_basis * rand_change
            psi = normalise(psi, dr)

    print("Final Energy:", energy(psi, V, dr))
    t2 = time.time()
    print("The time for the " + str(n) + "th iteration is:", t2 - t1, "s.\n")

    # Correction of artifacts at edge:
    if fix_artifacts:
        for ax in range(num_axes):
            for j in range(len(previous_states)):
                psi[ax, j] = 0
        psi = normalise(psi, dr)

    for ax in range(num_axes):
        plt.plot(r[ax], psi[ax])
        plt.title("The {0}th State for the Harmonic Oscillator along <{1}>:".format(n, ax))
        plt.ylabel("$\psi$")
        plt.xlabel("$r_{}$".format(ax))
        plt.show()

    return psi


def main():
    a, b, num_axes, N, num_iterations = -5, 5, 2, 100, 10 ** 5
    x = np.linspace(a, b, N)
    r = np.empty((num_axes, N))
    for ax in range(num_axes):
        r[ax] = x.copy()

    dr = (b - a) / N
    generate_derivative_matrix(N, dr)
    existing_states = np.zeros((num_axes, N))
    number_states = 5
    for i in range(number_states):
        psi = nth_state(a, b, num_axes, N, num_iterations, existing_states)

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
