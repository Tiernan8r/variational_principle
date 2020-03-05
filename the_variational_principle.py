import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# global constants:
hbar = 6.582119569 * 10 ** -16  # 6.582119569x10^-16 (from wikipedia)
# electron
m = 9.1093826 * 10 ** -31  # 9.1093837015(28)x10^-31
# for alpha particle:
# m = 2 * 1.67492728 * 10 ** -27 + 2 * 1.67262171 * 10 ** -27  # 9.1093837015(28)x10^-31
factor = -(hbar ** 2) / (2 * m)

pi = np.pi
e_0 = 8.854187817 * 10 ** -12
Z = 82
e = 1.60217653 * 10 ** -19


def normalise(psi: np.ndarray, dx: float):
    # integrate using the rectangular rule
    norm = np.sum(psi * psi) * dx
    norm_psi = psi / np.sqrt(norm)
    return norm_psi


global A
axes = ("x", "y", "z", "w", "q", "r", "s", "t", "u", "v")
pot_sys_name = "Linear Harmonic Oscillator"
colour_map = "hsv"


def generate_derivative_matrix(axis_length: int, dx: float):
    global A
    A = np.zeros((axis_length, axis_length))
    for i in range(1, axis_length - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = 1, -2, 1
    # forward & backward difference at the edges
    A[0, 0], A[0, 1], A[0, 2], A[-1, -1], A[-1, -2], A[-1, -3] = 1, -2, 1, 1, -2, 1
    return A * (dx ** -2)


def energy(psi: np.ndarray, V: np.ndarray, dx: float):
    # when V is inf, wil get an invalid value error at runtime, not an issue, is sorted in filtering below:
    Vp = V * psi
    # filter out nan values in Vp
    Vp = np.where(np.isfinite(Vp), Vp, 0)
    # A is the 2nd derivative matrix.
    Tp = factor * (A @ psi)

    return np.sum(psi * (Tp + Vp)) * dx


def potential(x: np.ndarray):
    # length = len(x)
    # third = length // 3
    # # mid, bef = numpy.zeros(third + 1), numpy.linspace(numpy.inf, numpy.inf, third)
    # mid, bef = numpy.zeros(third + 1), numpy.linspace(10, 10, third)
    # aft = bef.copy()
    # return numpy.concatenate((bef, mid, aft))

    return 0.5 * x ** 2


def gen_orthonormal_states(pre_existing_states: np.ndarray, axis_size, fix_artifacts=True):
    # there are no known states already
    if pre_existing_states.size == 0:
        return np.identity(axis_size)
    else:

        orthonormal_states = la.null_space(pre_existing_states)
        n = len(pre_existing_states)

        # artifacts fix
        if fix_artifacts:
            for j in range(n):
                for k in range(len(orthonormal_states[n])):
                    orthonormal_states[j, k] = 0

        return orthonormal_states.T


def nth_state(start: float, stop: float, axis_length: int, num_iterations: int,
              prev_psi: np.ndarray,
              fix_infinities=True, fix_artifacts=True, include_potential=False, plot_scale=10):
    n = len(prev_psi)

    t1 = time.time()
    # TODO error in inf occurs because null_space returned is wrong?
    #  occurs because 1st state == 0th state => orthonormals goosed.
    #  therefore: make 1 good -> all good?

    orthonormal_states = gen_orthonormal_states(prev_psi, axis_length)
    num_columns = orthonormal_states.shape[0]

    random.seed("THE-VARIATIONAL-PRINCIPLE")

    dx = (stop - start) / axis_length

    x = np.linspace(start, stop, axis_length)

    V = potential(x)

    psi = np.ones(axis_length)
    psi[0], psi[-1] = 0, 0

    if fix_infinities:
        # handling for the inf values in the infinite square well, or similar:
        for j in range(len(psi)):
            if not np.isfinite(V[j]):
                psi[j] = 0

        # infinite fix
        for k in range(len(psi)):
            if not np.isfinite(V[k]):
                for j in range(len(orthonormal_states)):
                    orthonormal_states[j, k] = 0

    psi = normalise(psi, dx)

    prev_E = energy(psi, V, dx)
    print("Initial Energy:", prev_E)

    for i in range(num_iterations):
        rand_index = random.randrange(1, num_columns - 1)

        rand_change = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_change *= -1

        orthonormal_basis = orthonormal_states[rand_index]

        psi += orthonormal_basis * rand_change
        psi = normalise(psi, dx)

        new_E = energy(psi, V, dx)
        if new_E < prev_E:
            prev_E = new_E
        else:
            psi -= orthonormal_basis * rand_change
            psi = normalise(psi, dx)

    print("Final Energy:", energy(psi, V, dx))
    t2 = time.time()
    print("The time for the " + str(n) + "th iteration is:", t2 - t1, "s.\n")

    # Correction of artifacts at edge:
    if fix_artifacts:
        for j in range(n + 1):
            psi[j] = 0
        psi = normalise(psi, dx)

    if include_potential:
        plt.plot(x, V)
        plt.plot(x, psi * plot_scale)
        plt.legend(("Potential", "{}th State".format(n)))
    else:
        plt.plot(x, psi)
    plt.title("The {}th State for the {} along ${}$".format(n, pot_sys_name, "x"))
    plt.xlabel("${}$".format("x"))
    plt.ylabel("$\psi$")
    plt.show()

    return psi


def main():
    a, b, N = -10, 10, 100
    num_states = 2
    num_iterations = num_states * 10 ** 5

    include_potential = False
    potential_scaling = 10

    x = np.linspace(a, b, N)

    V = potential(x)

    dx = (b - a) / N

    generate_derivative_matrix(N, dx)
    all_psi = np.zeros((1, N))

    for i in range(num_states):
        psi = nth_state(a, b, N, num_iterations, all_psi, include_potential=include_potential,
                        plot_scale=potential_scaling)

        all_psi = np.vstack((all_psi, psi))

    scale = 1
    if include_potential:
        scale = potential_scaling
        plt.plot(x, V)

    for n in range(len(all_psi)):
        plt.plot(x, all_psi[n] * scale)

    plt.title("Wavefunctions $\psi$ for the {} along ${}$:".format(pot_sys_name, "x"))
    plt.xlabel("${}$".format(a))
    plt.ylabel("$\psi$")
    if not include_potential:
        plt.legend(("Ground State", "Second State", "Third State", "Fourth State", "..."))
    else:
        plt.legend(("Potential", "Ground State", "Second State", "Third State", "Fourth State", "..."))
    plt.show()


main()
