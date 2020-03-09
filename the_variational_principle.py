import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as spr

# global constants:
hbar = 6.582119569 * 10 ** -16  # 6.582119569x10^-16 (from wikipedia)
# electron
m = 9.1093826 * 10 ** -31  # 9.1093837015(28)x10^-31
factor = -(hbar ** 2) / (2 * m)

global DEV2
axes = ("x", "y", "z", "w", "q", "r", "s", "t", "u", "v")
pot_sys_name = "Linear Harmonic Oscillator"
colour_map = "hsv"


def normalise(psi: np.ndarray, dr: float, num_axes: int):
    # integrate using the rectangular rule
    norm = np.sum(psi * psi) * (dr ** num_axes)
    norm_psi = psi / np.sqrt(norm)
    return norm_psi


def dev_mat(num_axes: int, N: int, axis_number: int):
    # missing the 1/dr^2 factor

    # cap axis_number in range to prevent errors.
    axis_number %= num_axes

    num_repeats = num_axes - (axis_number + 1)
    # The general pattern for a derivative matrix along the axis: axis_number, for a num_axes number of
    # dimensions, each of length N
    diagonals = [[-2] * N ** num_axes,
                 (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_repeats,
                 (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_repeats]
    D = spr.diags(diagonals, [0, -N ** axis_number, N ** axis_number], shape=(N ** num_axes, N ** num_axes))
    return D


def gen_DEV2(num_axes: int, axis_length: int, dr: float):
    global DEV2

    DEV2 = None

    for ax in range(num_axes):
        D = dev_mat(num_axes, axis_length, ax)
        if DEV2 is None:
            DEV2 = D
        else:
            DEV2 += D

    DEV2 *= (dr ** -2)


def energy(psi: np.ndarray, V: np.ndarray, dx: float):
    # when V is inf, wil get an invalid value error at runtime, not an issue, is sorted in filtering below:
    Vp = V * psi
    # filter out nan values in Vp
    Vp = np.where(np.isfinite(Vp), Vp, 0)
    # A is the 2nd derivative matrix.
    Tp = factor * (DEV2 @ psi)

    # TODO sum may need to change for n dimensions.
    return np.sum(psi * (Tp + Vp)) * (dr ** num_axes)


def potential(r: np.ndarray):
    # length = len(x)
    # third = length // 3
    # # mid, bef = numpy.zeros(third + 1), numpy.linspace(numpy.inf, numpy.inf, third)
    # mid, bef = numpy.zeros(third + 1), numpy.linspace(10, 10, third)
    # aft = bef.copy()
    # return numpy.concatenate((bef, mid, aft))

    return np.sum(0.5 * r ** 2, axis=0)


def gen_orthonormal_states(pre_existing_states: np.ndarray, axis_size, fix_artifacts=True) -> np.ndarray:
    # TODO needs nD TLC.

    # there are no known states already
    if pre_existing_states.size == 0:
        return spr.identity(axis_size)
    else:

        orthonormal_states = la.null_space(pre_existing_states)

        # # artifacts fix
        # n = len(pre_existing_states)
        # if fix_artifacts:
        #     for j in range(n):
        #         for k in range(len(orthonormal_states[n])):
        #             orthonormal_states[j, k] = 0

        return orthonormal_states.T


def nth_state(r: np.ndarray, dr: float, num_axes: int, axis_length: int, num_iterations: int,
              prev_psi_linear: np.ndarray,
              fix_infinities=True, fix_artifacts=True, include_potential=False, plot_scale=10) -> np.ndarray:
    n = len(prev_psi_linear)

    t1 = time.time()
    # TODO error in inf occurs because null_space returned is wrong?
    #  occurs because 1st state == 0th state => orthonormals goosed.
    #  therefore: make 1 good -> all good?

    # orthonormal_states = gen_orthonormal_states(prev_psi_linear, axis_length ** num_axes)
    orthonormal_states = la.null_space(prev_psi_linear).T

    num_columns = len(orthonormal_states)

    random.seed("THE-VARIATIONAL-PRINCIPLE")

    V = potential(r)
    V = V.reshape(axis_length ** num_axes)

    psi = np.ones([axis_length] * num_axes)
    # The Boundary Conditions
    psi[0], psi[-1] = 0, 0
    if num_axes > 1:
        for ax in range(1, axis_length - 1):
            row = psi[ax]
            col = row.T
            row[0], row[-1], col[0], col[-1] = 0, 0, 0, 0

    # linearise psi
    psi = psi.reshape(axis_length ** num_axes)

    # if fix_infinities:
    #     # handling for the inf values in the infinite square well, or similar:
    #     for j in range(len(psi)):
    #         if not np.isfinite(V[j]):
    #             psi[j] = 0
    #
    #     # infinite fix
    #     for k in range(len(psi)):
    #         if not np.isfinite(V[k]):
    #             for j in range(len(orthonormal_states)):
    #                 orthonormal_states[j, k] = 0

    psi = normalise(psi, dr, num_axes)

    prev_E = energy(psi, V, dr, num_axes)
    print("Initial Energy:", prev_E)

    for i in range(num_iterations):
        rand_index = random.randrange(1, num_columns - 1)

        rand_change = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_change *= -1

        orthonormal_basis = orthonormal_states[rand_index]

        psi += orthonormal_basis * rand_change
        psi = normalise(psi, dr, num_axes)

        new_E = energy(psi, V, dr, num_axes)
        if new_E < prev_E:
            prev_E = new_E
        else:
            psi -= orthonormal_basis * rand_change
            psi = normalise(psi, dr, num_axes)

    print("Final Energy:", energy(psi, V, dr, num_axes))
    t2 = time.time()
    print("The time for the " + str(n) + "th iteration is:", t2 - t1, "s.\n")

    # # Correction of artifacts at edge:
    # if fix_artifacts:
    #     for j in range(n + 1):
    #         psi[j] = 0
    #     psi = normalise(psi, dx)

    # # TODO fix this mess
    # if num_axes == 1:
    #     if include_potential:
    #         plt.plot(r[0], V)
    #         plt.plot(r[0], psi * plot_scale)
    #         plt.legend(("Potential", "{}th State".format(n)))
    #     else:
    #         plt.plot(r[0], psi)
    #     plt.title("The {}th State for the {} along ${}$".format(n, pot_sys_name, "x"))
    #     plt.xlabel("${}$".format("x"))
    #     plt.ylabel("$\psi$")
    #     plt.show()

    return psi.reshape([axis_length] * num_axes)


def plotting(r, all_psi, num_axes, include_V=False, V=None):
    cmap = plt.cm.get_cmap(colour_map)

    def plot_wireframe(x, y, z, title, zlabel):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot_wireframe(x, y, z)
        ax.set_zlabel(zlabel)
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    def plot_surface(x, y, z, title, zlabel):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(x, y, z, cmap=cmap)
        fig.colorbar(surf, ax=ax)
        ax.set_zlabel(zlabel)
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    def plot_img(x, y, z, title):
        plt.contourf(x, y, z, cmap=cmap)
        # plt.contourf(x, y, z)
        plt.colorbar()
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    def plot_line(x, y, title):
        plt.plot(x, y)
        plt.xlabel("$x$")
        plt.ylabel("$\psi$")
        plt.title(title)
        plt.show()

    def plot_3D_scatter(x, y, z, vals, title):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        p = ax.scatter3D(x, y, zs=z, c=vals, cmap=colour_map)
        fig.colorbar(p, ax=ax)

        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ax.set_zlabel("$z$")
        plt.show()

    if num_axes == 1:

        num_extra = len(all_psi) - 1

        # legend = ["Ground State", "1st State", "2nd State", "3rd State", "4th State"]
        legend = ["Ground State"]
        for i in range(num_extra):
            th = "th"
            if i == 1:
                th = "st"
            elif i == 2:
                th = "nd"
            elif i == 3:
                th = "rd"
            legend += ["{}{} State".format(i, th)]

        if include_V:
            all_psi = np.vstack((V, all_psi))
            legend = ["Potential"] + legend
        for i in range(len(all_psi)):
            title = "The {} for the {} along $x$:".format(legend[i], pot_sys_name)
            plot_line(*r, all_psi[i], title)

    elif num_axes == 2:

        if include_V:
            title = "The Potential function for the {} along $x$ & $y$".format(pot_sys_name)
            plot_img(*r, V, title)
            plot_wireframe(*r, V, title, "V")
            plot_surface(*r, V, title, "V")

        num_states = len(all_psi)
        for n in range(num_states):
            title = "$\psi_{}$ for the {} along $x$ & $y$".format(n, pot_sys_name)
            plot_img(*r, all_psi[n], title)
            plot_wireframe(*r, all_psi[n], title, "$\psi$")
            plot_surface(*r, all_psi[n], title, "$\psi$")

    elif num_axes == 3:

        if include_V:
            title = "The Potential function for the {} along $x$, $y$ & $z$".format(pot_sys_name)
            plot_3D_scatter(*r, V, title)

        num_states = len(all_psi)
        for n in range(num_states):
            title = "$\psi_{}$ for the {} along $x$, $y$ & $z$".format(n, pot_sys_name)
            plot_3D_scatter(*r, all_psi[n], title)

    else:
        return

    # if num_axes == 2:
    #
    #     if include_V:
    #         title = "The Potential function for {} along $x$ & $y$".format(pot_sys_name)
    #         plot_img(*r, V, title)
    #         plot_wireframe(*r, V, title, "V")
    #         plot_surface(*r, V, title, "V")
    #
    #     num_states = len(all_psi)
    #     for n in range(num_states):
    #         title = "$\psi_{}$ for the {} along $x$ & $y$".format(n, pot_sys_name)
    #         plot_img(*r, all_psi[n], title)
    #         plot_wireframe(*r, all_psi[n], title, "$\psi$")
    #         plot_surface(*r, all_psi[n], title, "$\psi$")
    # elif num_axes == 1:
    #
    #     legend = ["Ground State", "1st State", "2nd State", "3rd State", "4th State"]
    #     if include_V:
    #         all_psi = [V] + all_psi
    #         legend = ["Potential"] + legend
    #     for i in range(len(all_psi)):
    #         title = "The {} for the {} along $x$:".format(legend[i], pot_sys_name)
    #         plot_line(*r, all_psi[i], title)
    #
    # else:
    #     return


def main():
    # TODO overhaul

    # initially is symmetric grid.
    a, b, N = -10, 10, 100
    num_states = 1
    num_axes = 1
    num_iterations = 10 ** 5

    include_potential = True
    potential_scaling = 100

    x = np.linspace(a, b, N)
    tmp_r = [x] * num_axes
    r = np.array(np.meshgrid(*tmp_r, indexing="ij"))

    V = potential(r)

    dr = (b - a) / N

    gen_DEV2(num_axes, N, dr)
    # all_psi stores each nth state of psi as a list of the psi states.
    all_psi_linear = np.zeros([1] + [N ** num_axes])
    all_psi = np.zeros([1] + [N] * num_axes)

    for i in range(num_states):
        psi = nth_state(r, dr, num_axes, N, num_iterations, all_psi_linear, include_potential=include_potential,
                        plot_scale=potential_scaling)

        all_psi = np.vstack((all_psi, [psi]))

        psi_linear = psi.reshape(N ** num_axes)
        all_psi_linear = np.vstack((all_psi_linear, [psi_linear]))
    all_psi = all_psi[1:]
    all_psi_linear = all_psi_linear[1:]

    # # TODO fix this hack fest
    # if num_axes == 1:
    #     scale = 1
    #     if include_potential:
    #         scale = potential_scaling
    #         plt.plot(x, V)
    #
    #     for n in range(len(all_psi_linear)):
    #         plt.plot(x, all_psi_linear[n] * scale)
    #
    #     plt.title("Wavefunctions $\psi$ for the {} along ${}$:".format(pot_sys_name, "x"))
    #     plt.xlabel("${}$".format(a))
    #     plt.ylabel("$\psi$")
    #     legend = ["Ground State", "Second State", "Third State", "Fourth State", "..."]
    #     if include_potential:
    #         legend = ["Potential"] + legend
    #     plt.legend(legend)
    #     plt.show()

    plotting(r, all_psi, num_axes, include_potential, V)


main()
