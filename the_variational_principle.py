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


def normalise(psi: np.ndarray, dr: float):
    # integrate using the rectangular rule
    # norm = np.sum(psi * psi) * dr
    norm = 0
    psi_sq = psi * psi
    for ax in range(len(psi_sq)):
        norm += np.sum(psi_sq[ax])
    norm *= dr
    norm_psi = psi / np.sqrt(norm)
    return norm_psi


global A
axes = ("x", "y", "z", "w", "q", "r", "s", "t", "u", "v")
pot_sys_name = "Linear Harmonic Oscillator"
colour_map = "hsv"


def generate_derivative_matrix(axis_length: int, dr: float):
    global A
    A = np.zeros((axis_length, axis_length))
    for i in range(1, axis_length - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = 1, -2, 1
    # forward & backward difference at the edges
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
    V = []
    for ax in range(len(r)):
        # axis_length = len(r[ax])
        # third = axis_length // 3
        # # mid, bef = np.zeros(third + 1), np.linspace(np.inf, np.inf, third)
        # mid, bef = np.zeros(third + 1), np.linspace(10, 10, third)
        # aft = bef.copy()
        # V += [np.concatenate((bef, mid, aft))]

        # B_c = 10
        # U = -15
        #
        # axis_length = len(r[ax])
        # third = axis_length // 3
        # well = np.concatenate(([10**5], np.linspace(U, U, third), [B_c]))
        # r_aft = r[ax][third + 2:]
        # coulomb = (Z - 2) * 2 * e ** 2 / (4 * pi * e_0 * r_aft)
        # V += [np.concatenate((well, coulomb))]

        V += [0.5 * r[ax] ** 2]
        # V += [1 / r[ax]]
    return np.array(V)


def gen_orthonormal_states(pre_existing_states: np.ndarray, num_axes, axis_size, fix_artifacts=True):
    # there are no known states already
    if pre_existing_states.size == 0:
        identity = np.empty((num_axes, axis_size, axis_size))
        for ax in range(num_axes):
            identity[ax] = np.identity(axis_size)
        return identity
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
              fix_infinities=True, fix_artifacts=True, include_potential=False, plot_scale=10):
    n = (len(previous_states) // num_axes) - 1

    t1 = time.time()
    # TODO error in inf occurs because null_space returned is wrong?
    #  occurs because 1st state == 0th state => orthonormals goosed.
    #  therefore: make 1 good -> all good?

    orthonormal_states = gen_orthonormal_states(previous_states, num_axes, axis_length)
    num_columns = len(orthonormal_states)

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
                        orthonormal_states[j, k] = 0

    psi = normalise(psi, dr)

    prev_E = energy(psi, V, dr)
    print("Initial Energy:", prev_E)

    for i in range(num_iterations):
        rand_index = random.randrange(1, num_columns - 1)

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
        a = axes[ax]
        if include_potential:
            plt.plot(r[ax], V[ax])
            plt.plot(r[ax], psi[ax] * plot_scale)
            plt.legend(("Potential", "{}th State".format(n)))
        else:
            plt.plot(r[ax], psi[ax])
        plt.title("The {}th State for the {} along ${}$".format(n, pot_sys_name, a))
        plt.xlabel("${}$".format(a))
        plt.ylabel("$\psi$")
        plt.show()

    return psi


def plotting(r, all_psi, num_axes, include_V=False, V=None):
    def plot_img(x, y, z, title):
        # cmap = plt.cm.get_cmap(colour_map)

        # plt.contourf(x, y, z, cmap=cmap)
        plt.contourf(x, y, z)
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    if num_axes == 2:
        x = r[0]
        y = r[1]

        XX, YY = np.meshgrid(x, y)

        if include_V:
            V_x = V[0]
            V_y = V[1]

            V_XX, V_YY = np.meshgrid(V_x, V_y)
            Z = V_XX + V_YY

            title = "The Potential function for {} along $x$ & $y$".format(pot_sys_name)
            plot_img(XX, YY, Z, title)

        num_states = (len(all_psi) // num_axes) - 1
        for n in range(num_states):
            psi_x = all_psi[2 * (n + 1)]
            psi_y = all_psi[2 * (n + 1) + 1]

            psi_XX, psi_YY = np.meshgrid(psi_x, psi_y)
            Z = psi_XX + psi_YY

            title = "$\psi_{}$ for the {} along $x$ & $y$".format(n, pot_sys_name)
            plot_img(XX, YY, Z, title)

    else:
        return


def main():
    a, b, num_axes, N = -10, 10, 2, 100
    num_states = 10
    num_iterations = 5 * 10 ** 4

    include_potential = False
    potential_scaling = 10

    x = np.linspace(a, b, N)
    r = np.empty((num_axes, N))
    for ax in range(num_axes):
        r[ax] = x.copy()

    V = potential(r)

    dr = (b - a) / N

    generate_derivative_matrix(N, dr)
    all_psi = np.zeros((num_axes, N))
    # group the psi wavefunctions according to their axes, ie all the x values together, y together, etc...
    psi_by_axis = []

    for i in range(num_states):
        psi = nth_state(a, b, num_axes, N, num_iterations, all_psi, include_potential=include_potential,
                        plot_scale=potential_scaling)

        all_psi = np.vstack((all_psi, psi))

        if len(psi_by_axis) == 0:
            tmp_psi_by_axis = []
            for ax in range(num_axes):
                tmp_psi_by_axis += [[psi[ax].copy()]]
            psi_by_axis = tmp_psi_by_axis.copy()
        else:
            tmp_psi_by_axis = []
            for ax in range(num_axes):
                tmp_psi_by_axis += [np.append(psi_by_axis[ax], [psi[ax]], axis=0)]
            psi_by_axis = tmp_psi_by_axis.copy()

    for ax in range(num_axes):

        scale = 1
        if include_potential:
            scale = potential_scaling
            plt.plot(r[ax], V[ax])

        for n in range(len(psi_by_axis[ax])):
            plt.plot(r[ax], psi_by_axis[ax][n] * scale)

        a = axes[ax]
        plt.title("Wavefunctions $\psi$ for the {} along ${}$:".format(pot_sys_name, a))
        plt.xlabel("${}$".format(a))
        plt.ylabel("$\psi$")
        if not include_potential:
            plt.legend(("Ground State", "Second State", "Third State", "Fourth State", "..."))
        else:
            plt.legend(("Potential", "Ground State", "Second State", "Third State", "Fourth State", "..."))
        plt.show()

    plotting(r, all_psi, num_axes)


main()
