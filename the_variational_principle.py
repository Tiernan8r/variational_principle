import random
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy.linalg as la
import scipy.sparse as spr

# global constants:
hbar = 6.582119569 * 10 ** -16  # 6.582119569x10^-16 (from wikipedia)
# electron
m = 9.1093826 * 10 ** -31  # 9.1093837015(28)x10^-31
factor = -(hbar ** 2) / (2 * m)

th = {1: "st", 2: "nd", 3: "rd"}
global DEV2
axes = ("x", "y", "z", "w", "q", "r", "s", "t", "u", "v")
pot_sys_name = "Infinite Square Well"
colour_map = "autumn"


def normalise(psi: np.ndarray, dr: float) -> np.ndarray:
    # integrate using the rectangular rule
    norm = (psi * psi).sum() * dr
    norm_psi = psi / np.sqrt(norm)
    return norm_psi


def boundary_conditions(f: np.ndarray, N: int, D: int) -> np.ndarray:
    # The Boundary Conditions
    f[0], f[-1] = 0, 0
    if D > 1:
        for ax in range(1, N - 1):
            row = f[ax]
            col = row.T
            row[0], row[-1], col[0], col[-1] = 0, 0, 0, 0
    return f


def dev_mat(D: int, N: int, axis_number: int, dr: float) -> np.ndarray:
    # cap axis_number in range to prevent errors.
    axis_number %= D

    num_repeats = D - (axis_number + 1)
    # The general pattern for a derivative matrix along the axis: axis_number, for a num_axes number of
    # dimensions, each of length N
    diagonals = [[-2] * N ** D,
                 (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_repeats,
                 (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_repeats]
    D_n = spr.diags(diagonals, [0, -N ** axis_number, N ** axis_number], shape=(N ** D, N ** D))

    return D_n * (dr ** -2)


def gen_DEV2(D: int, N: int, dr: float):
    global DEV2

    DEV2 = None

    for ax in range(D):
        D_n = dev_mat(D, N, ax, dr)
        if DEV2 is None:
            DEV2 = D_n
        else:
            DEV2 += D_n


def energy(psi: np.ndarray, V: np.ndarray, dr: float) -> float:
    # when V is inf, wil get an invalid value error at runtime, not an issue, is sorted in filtering below:
    Vp = V * psi
    # filter out nan values in Vp
    Vp = np.where(np.isfinite(Vp), Vp, 0)
    # A is the 2nd derivative matrix.
    Tp = factor * (DEV2 @ psi)

    return (psi * (Tp + Vp)).sum() * dr


def potential(r: np.ndarray) -> np.ndarray:
    V = r.copy()
    for i in range(len(r)):
        length = len(r[i])
        third = length // 3

        addition = int(abs((third - (length / 3))) * 3)

        mid, bef = np.zeros(third + addition), np.linspace(np.inf, np.inf, third)
        # mid, bef = np.zeros(third + addition), np.linspace(10, 10, third)
        aft = bef.copy()
        # return np.concatenate((bef, mid, aft))
        V[i] = np.concatenate((bef, mid, aft))

    # V = np.zeros(r.shape)
    # V = np.sum(0.5 * r ** 2, axis=0)
    return V


def nth_state(r: np.ndarray, dr: float, D: int, N: int, num_iterations: int,
              prev_psi_linear: np.ndarray, n: int) -> np.ndarray:
    t1 = time.time()

    orthonormal_basis = la.null_space(prev_psi_linear).T

    for j in range(len(orthonormal_basis.T)):
        plt.plot(orthonormal_basis.T[j])
    # plt.plot(orthonormal_basis)
    plt.title("RETURNED ORTHONORMAL BASIS")
    plt.show()

    random.seed("THE-VARIATIONAL-PRINCIPLE")

    V = potential(r)
    V = V.reshape(N ** D)

    # psi = np.empty([N] * D)
    psi = 0.5 * r ** 2

    # psi = boundary_conditions(psi, N, D)

    # linearise psi
    psi = psi.reshape(N ** D)

    len_V = len(V)
    nan_indices = [False] * len_V
    for j in range(len_V):
        if not np.isfinite(V[j]):
            a, b = j - 1, j + 1
            if a < 0:
                a = 0
            if b >= len_V:
                b = len_V - 1

            nan_indices[a] = nan_indices[j] = nan_indices[b] = True

    psi = np.where(nan_indices, 0, psi)

    for j in range(n - 1):
        nan_indices[j] = False

    # for j in range(len(orthonormal_basis.T)):
    #     plt.plot(orthonormal_basis.T[j])
    #     plt.title("OB" + str(j))
    #     plt.show()
    orthonormal_basis = np.where(nan_indices, 0, orthonormal_basis)

    # # Handling of infinity values in psi and corresponding entries in orthonormal_basis
    # for j in range(len(psi)):
    #     if not np.isfinite(V[j]) or not np.isfinite(psi[j]):
    #         psi[j] = 0
    #
    # print("SHAPE OF OB:", orthonormal_basis.shape)
    # # infinite fix of the orthonormal basis
    # for j in range(len(orthonormal_basis)):
    #     for k in range(len(V)):
    #         if not np.isfinite(V[k]):
    #             orthonormal_basis[j, k] = 0
    #
    for j in range(len(orthonormal_basis.T)):
        plt.plot(orthonormal_basis.T[j])
    # plt.plot(orthonormal_basis)
    plt.title("INFINITE FIXED OB")
    plt.show()

    prev_E = energy(psi, V, dr)
    print("Initial Energy:", prev_E)

    num_bases = len(orthonormal_basis)
    for i in range(num_iterations):

        # keep_looping = True
        # while keep_looping:
        #     rand_index = random.randrange(num_bases)
        #     keep_looping = not np.isfinite(V[rand_index])
        rand_index = random.randrange(num_bases)

        rand_change = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_change *= -1

        basis_vector = orthonormal_basis[rand_index]

        psi += basis_vector * rand_change
        psi = normalise(psi, dr)

        new_E = energy(psi, V, dr)
        if new_E < prev_E:
            prev_E = new_E
        else:
            psi -= basis_vector * rand_change
            psi = normalise(psi, dr)

    print("Final Energy:", energy(psi, V, dr))
    t2 = time.time()
    print("The time for the " + str(n) + "th iteration is:", t2 - t1, "s.\n")

    psi = psi.reshape([N] * D)

    # Correction of phase, to bring it to the positive for nicer plotting.
    phase = np.sum(psi) * dr
    if phase < 0:
        psi *= -1

    return psi


def plotting(r, all_psi, D, include_V=False, V=None):
    cmap = plt.cm.get_cmap(colour_map)

    def plot_wireframe(x, y, z, title, zlabel="$\psi$"):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot_wireframe(x, y, z)
        ax.set_zlabel(zlabel)
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    def plot_surface(x, y, z, title, zlabel="$\psi$"):
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

    def plot_line(x, y, title, ylabel="$\psi$", filename=None):
        plt.plot(x, y)
        plt.xlabel("$x$")
        plt.ylabel(ylabel)
        plt.title(title)
        if filename is not None:
            plt.savefig(filename)
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

    if D == 1:

        num_extra = len(all_psi)

        legend = ["Ground State"]
        for i in range(1, num_extra):
            legend += ["{}{} State".format(i, th.get(i, "th"))]

        if include_V:
            all_psi = np.vstack((V, all_psi))
            legend = ["Potential"] + legend
        for i in range(len(all_psi)):
            title = "The {} for the {} along $x$:".format(legend[i], pot_sys_name)
            fname = "state_{}".format(i - 1)
            state = "$\psi$"
            if legend[i] == "Potential":
                fname = "potential"
                state = "V"
            plot_line(*r, all_psi[i], title, state, fname)

    elif D == 2:

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

    elif D == 3:

        if include_V:
            title = "The Potential function for the {} along $x$, $y$ & $z$".format(pot_sys_name)
            plot_3D_scatter(*r, V, title)

        num_states = len(all_psi)
        for n in range(num_states):
            title = "$\psi_{}$ for the {} along $x$, $y$ & $z$".format(n, pot_sys_name)
            plot_3D_scatter(*r, all_psi[n], title)

    else:
        return


def main():
    # initially is symmetric grid.
    a, b, N = -10, 10, 100
    num_states = 3
    if num_states >= N:
        num_states = N - 2

    D = 1
    # Iteration recommendations:
    # 10**5 for 1D
    # 10**6 for 2D
    # 10**7 for 3D (loooong)
    num_iterations = 10 ** 5

    include_potential = True
    potential_scaling = 100

    x = np.linspace(a, b, N)
    tmp_r = [x] * D
    r = np.array(np.meshgrid(*tmp_r, indexing="ij"))

    V = potential(r)

    dr = (b - a) / N

    gen_DEV2(D, N, dr)
    # all_psi stores each nth state of psi as a list of the psi states.
    initially_empty = True
    all_psi_linear = np.zeros([1] + [N ** D])
    all_psi = np.zeros([1] + [N] * D)

    for i in range(num_states):
        psi = nth_state(r, dr, D, N, num_iterations, all_psi_linear, i + 1)

        psi_linear = psi.reshape(N ** D)
        if initially_empty:
            all_psi_linear = np.array([psi_linear])
            all_psi = np.array([psi])
            initially_empty = False
        else:
            all_psi_linear = np.vstack((all_psi_linear, [psi_linear]))
            all_psi = np.vstack((all_psi, [psi]))

    plotting(r, all_psi, D, include_potential, V)


main()
