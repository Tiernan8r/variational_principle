import random
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy.linalg as la
import scipy.sparse as spr

# global constants:
hbar = 6.5821189 * 10 ** -16  # 6.582119569x10^-16 eV (from wikipedia)
# electron
m = 9.1093819 * 10 ** -31  # 9.1093837015(28)x10^-31
# factor used in calculation of energy
factor = -(hbar ** 2) / (2 * m)

# neatness nicety, for displaying indexing of the states.
th = {1: "st", 2: "nd", 3: "rd"}

# The Lagrangian Derivative matrix
global DEV2
# a list of names for the 1st 10 axes.
axes = ("x", "y", "z", "w", "q", "r", "s", "t", "u", "v")
# The name of the QM system to be shown in the plot titles.
pot_sys_name = "Harmonic Oscillator"
# the colour map for the surface plots.
colour_map = "autumn"


def normalise(psi: np.ndarray, dr: float) -> np.ndarray:
    """
    The function takes in a non-normalised psi wavefunction, and returns the normalised version of it.
    :param psi: The wavefunction to normalise.
    :param dr: The grid spacing of the wavefunction.
    :return: The normalised wavefunction
    """
    # integrate using the rectangular rule
    norm = (psi * psi).sum() * dr
    # Since psi is displayed as |psi|^2, take the sqrt of the norm
    norm_psi = psi / np.sqrt(norm)
    return norm_psi


def dev_mat(D: int, N: int, axis_number: int, dr: float) -> np.ndarray:
    """
    Generates the sparse second derivative central difference derivative matrix along given axis, for a grid of dimensions N^D.
    :param D: The number of dimensions of the system, e.g.: 3D...
    :param N: The dimensions of the symmetric grid.
    :param axis_number: The axis to derive along, starting at 0.
    :param dr: The grid spacing in the system.
    :return: The second order central difference derivative sparse matrix along the given axis.
    """
    # cap axis_number in range to prevent errors.
    axis_number %= D

    # Determine the number of the cell grids that need to be repeated along to populate the matrix
    num_cells = D - (axis_number + 1)
    # The general pattern for a derivative matrix along the axis: axis_number, for a num_axes number of
    # dimensions, each of length N
    diagonals = [[-2] * N ** D,
                 (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_cells,
                 (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_cells]
    # Create a sparse matrix for the given diagonals, of the desired size.
    D_n = spr.diags(diagonals, [0, -N ** axis_number, N ** axis_number], shape=(N ** D, N ** D))

    # return the matrix, factored by the grid spacing as required by the central difference formula
    return D_n * (dr ** -2)


def generate_laplacian(D: int, N: int, dr: float):
    """
    Generates the Lagrangian second derivative matrix for the number of axes D.
    :param D: The number of dimensions/axes in the system.
    :param N: The size of each dimension.
    :param dr: The grid spacing in the system.
    """
    global DEV2

    # Initially set DEV2 to be undefined.
    DEV2 = None

    # iterate over each dimension in the system.
    for ax in range(D):
        # generate the second order central difference matrix for this axis
        D_n = dev_mat(D, N, ax, dr)
        # if it's the first matrix generated, set DEV2 equal to it.
        if DEV2 is None:
            DEV2 = D_n
        # otherwise add it, as matrix multiplication is distributive (ie differentiation is distributive)
        else:
            DEV2 += D_n


def energy(psi: np.ndarray, V: np.ndarray, dr: float) -> float:
    """
    Calculates the energy eigenvalue of a given wavefunction psi in a given potential system V.
    :param psi: The wavefunction in the system.
    :param V: The potential function of the system.
    :param dr: The grid spacing in the system.
    :return: The energy eigenvalue E.
    """
    # when V is inf, wil get an invalid value error at runtime, not an issue, is sorted in filtering below:
    Vp = V * psi
    # filter out nan values in Vp
    Vp = np.where(np.isfinite(Vp), Vp, 0)

    # Calculate the kinetic energy of the system
    # DEV2 is the lagrangian 2nd derivative matrix.
    Tp = factor * (DEV2 @ psi)

    # Return the integral of the KE and PE applied to psi, which is the energy.
    return (psi * (Tp + Vp)).sum() * dr


def potential(r: np.ndarray) -> np.ndarray:
    """
    The potential energy function of the system
    :param r: The coordinate grid of the system for each axis.
    :return: The potential function V as a grid of values for each position.
    """

    N = r.shape[1]
    D = r.shape[0]

    # # Infinite Sq Well
    # #V_0 = np.inf
    # # Finite Sq Well
    # V_0 = 10
    #
    # third = N // 3
    #
    # addition = int(abs((third - (N / 3))) * 3)
    #
    # mid, bef = np.zeros(third + addition), np.linspace(V_0, V_0, third)
    # aft = bef.copy()
    # x_well = np.concatenate((bef, mid, aft))
    # wells = [x_well] * D
    # V = np.array(np.meshgrid(*wells, indexing="ij"))
    #
    # # Correction of Corners:
    # V = np.sum(V, axis=0)
    # V = V.reshape(N ** D)
    # # Perturbation
    # R = np.sum(r, axis=0)
    # R = R.reshape(N ** D)
    # for i in range(len(V)):
    #     if V[i] > 0:
    #         V[i] = V_0
    #     # Perturbation
    #     elif V[i] == 0:
    #         V[i] += 0.5 * R[i]
    # V = V.reshape([N] * D)

    # # Alpha Barrier Penetration
    # V = r.copy()
    # for i in range(len(r)):
    #     length = len(r[i])
    #     third = length // 3
    #
    #     Z = 48
    #     e = 1.6021753*10**-19
    #     e_0 = 8.854187817*10**-12
    #     pi = np.pi
    #
    #     # # Rc = 1.2 A^1.3 fm
    #     # Rc = r[i][third]
    #     # A = (Rc / 1.2) ** (1/1.3) * 10**-15
    #
    #     # U = -np.array([2.4 * (Z - 2) * A **-(1/3)] * third)
    #     U = -np.array([0.05] * third)
    #     U[0] = np.inf
    #
    #     # inv_r = ((Z - 2) * 2 * e**2) / (4 * pi * e_0 * r[i][third:])
    #     inv_r = 1 / r[i][third:]
    #
    #     V[i] = np.concatenate((U, inv_r))
    # V = np.sum(V, axis=0)

    # Harmonic Oscillator:
    V = 0.5 * r ** 2
    V = np.sum(V, axis=0)

    # # Anharmonic Oscillator:
    # V = r + 0.5 * r ** 2 + 0.25 * r **4
    # V = np.sum(V, axis=0)

    # # V-shaped
    # V = abs(r)
    # V = np.sum(V, axis=0)

    # # Inverse R:
    # V = r ** -1
    # V = np.sum(V, axis=0)

    # # Inverse Sq Law
    # V = r ** -2
    # V = np.sum(V, axis=0)

    # # Free Particle
    # V = np.zeros(r.shape)
    # V = np.sum(V, axis=0)

    # # # Central Potential:
    # A = -10
    # B = 1.5
    # C = 8
    #
    # V_c = A * C * r**-1
    # V_f = B * C**2 * r**-2
    #
    # V = V_c + V_f
    # V = np.sum(V, axis=0)

    # # Crystal Band Structure:
    # num_bands = 5
    # band_spacing = N // (2 * num_bands)
    # V_0 = 10
    #
    # x_band = np.linspace(V_0, V_0, N)
    # # for i in range(1, 2 * num_bands, 2):
    # i = 0
    # j = -1
    # for k in range(N):
    #     i %= band_spacing
    #     j %= 2
    #     if j == 0:
    #         x_band[k] = 0
    #     if i == 0:
    #         j += 1
    #     i += 1
    #
    # # for i in range(num_bands + 2):
    # #     start_index = band_spacing * i * 2
    # #     for j in range(band_spacing):
    # #         x_band[start_index + j] = 0
    #
    # wells = [x_band] * D
    # V = np.array(np.meshgrid(*wells, indexing="ij"))
    #
    # # Correction of Corners:
    # V = np.sum(V, axis=0)
    # V = V.reshape(N ** D)
    # # # Perturbation
    # # R = np.sum(r, axis=0)
    # # R = R.reshape(N ** D)
    # for i in range(len(V)):
    #     if V[i] > 0:
    #         V[i] = V_0
    #     # # Perturbation
    #     # elif V[i] == 0:
    #     #     V[i] += 0.5 * R[i]
    # V = V.reshape([N] * D)

    # # Delta Barrier:
    # V = np.zeros(r.shape)
    # sub_V = V
    # while True:
    #     sub_sub_V = sub_V[0]
    #     if not type(sub_sub_V) is np.ndarray:
    #         L = len(sub_V)
    #         sub_V[L // 2] = -np.inf
    #         break
    #     else:
    #         sub_V = sub_sub_V
    # V = V.sum(axis=0)

    return V


def nth_state(r: np.ndarray, dr: float, D: int, N: int, num_iterations: int,
              prev_psi_linear: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the nth psi energy eigenstate wavefunction of a given potential system.
    :param r: The grid coordinates.
    :param dr: The grid spacing.
    :param D: The number of axes in the system.
    :param N: The size of each axis.
    :param num_iterations: The number of iterations to calculate over.
    :param prev_psi_linear: The previous calculated psi states for the potential system.
    :param n: The order of the state.
    :return: The energy eigenstate wavefunction psi of order n for the potential system.
    """
    # Get the time that calculations start at.
    t1 = time.time()

    # Get the orthonormal basis for this state, by finding the null space if the previous lower order psi
    orthonormal_basis = la.null_space(prev_psi_linear).T

    # Set a seed for repeatable results.
    random.seed("THE-VARIATIONAL-PRINCIPLE")

    # Calculate the potential of the system.
    V = potential(r)
    # turn the potential grid into a linear column vector for linear algebra purposes.
    V = V.reshape(N ** D)

    # generate an initial psi, I've found that a quadratic function works nicely (no discontinuities.)
    psi = (0.5 * r ** 2).sum(axis=0)
    # psi = np.ones(r.shape).sum(axis=0)

    # linearise psi from a grid to a column vector
    psi = psi.reshape(N ** D)

    # Account for infinite values in the potential:
    len_V = len(V)
    # Keep track of all the indices that have an inf value for the V.
    nan_indices = [False] * len_V
    for j in range(len_V):
        # # Tag the bordering points as well.
        # a, b = j - 1, j + 1
        # if a < 0:
        #     a = 0
        # if b >= len_V:
        #     b = len_V - 1
        #
        # if not np.isfinite(V[j]) and (not np.isfinite(V[a]) and not np.isfinite(V[b])):
        #     # nan_indices[a] = nan_indices[j] = nan_indices[b] = True
        #     nan_indices[j] = True
        if not np.isfinite(V[j]):
            nan_indices[j] = True

    # filter the corresponding psi values to be = 0
    psi = np.where(nan_indices, 0, psi)

    # filter the values in the orthonormal basis to be 0
    for j in range(n - 1):
        nan_indices[j] = False
    orthonormal_basis = np.where(nan_indices, 0, orthonormal_basis)

    # get a default initial energy to compare against.
    prev_E = energy(psi, V, dr)

    # Keep track of the number of orthonormal bases that there are.
    num_bases = len(orthonormal_basis)
    # loop for the desired number of iterations
    for i in range(num_iterations):

        # generate a random orthonormal basis to sample.
        rand_index = random.randrange(num_bases)

        # generate a random value to change by that converges to 0 as we sample more.
        rand_change = random.random() * 0.1 * (num_iterations - i) / num_iterations

        # 50% of the time, add, the other 50% take away
        if random.random() > 0.5:
            rand_change *= -1

        # get the orthonormal basis that we are sampling with
        basis_vector = orthonormal_basis[rand_index]

        # tweak the psi wavefunction by the generated change, with the given basis.
        psi += basis_vector * rand_change
        # re normalise the changed psi
        psi = normalise(psi, dr)

        # get the corresponding new energy for the changed psi
        new_E = energy(psi, V, dr)

        # if the new energy is lower than the current energy, keep the change.
        if new_E < prev_E:
            prev_E = new_E
        # otherwise set psi back to the way it was before the change.
        else:
            psi -= basis_vector * rand_change
            psi = normalise(psi, dr)

    # Display the final energy of the wavefunction to the console.
    print("Final Energy:", energy(psi, V, dr))
    # calculate how long the computation took.
    t2 = time.time()
    print("The time for the " + str(n) + "th iteration is:", t2 - t1, "s.\n")

    # turn psi back from a column vector to a grid.
    psi = psi.reshape([N] * D)

    # Correction of phase, to bring it to the positive for nicer plotting.
    phase = np.sum(psi) * dr
    if phase < 0:
        psi *= -1

    # return the generated psi as a grid.
    return psi


def plotting(r, all_psi, D, include_V=False, V=None, V_scale=1):
    """
    A method that contains various forms of plotting of psi and the potential of the system.
    :param r: The grid coordinates.
    :param all_psi: All the wavefunctions to plot.
    :param D: The number of axes in the system.
    :param include_V: Whether to plot the potential or not.
    :param V: The potential to plot if so.
    """
    # The colour map is set by the global string
    cmap = plt.cm.get_cmap(colour_map)

    # A method to plot the 1D system as a line.
    def plot_line(x, y, title, ylabel="$\psi$", legend=None, filename=None, include_V=False, V=None, V_scale=1):
        if include_V:
            plt.plot(x, V)
        plt.plot(x, y * V_scale)
        plt.xlabel("$x$")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(legend)
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    # A method to plot the 2D system as a flat image.
    def plot_img(x, y, z, title):
        plt.contourf(x, y, z, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    # A method to plot the 2D system as a wireframe.
    def plot_wireframe(x, y, z, title, zlabel="$\psi$"):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot_wireframe(x, y, z)
        ax.set_zlabel(zlabel)
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    # A method to plot the 2D system as a surface plot.
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

    # A method to plot the 3d system as a 3D scatter plot.
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

    # If the system is 1D, plot a line
    if D == 1:

        # The number of psi states to plot.
        num_states = len(all_psi)

        # Generate the legend for the system, naming the 0th state as the Ground State.
        state_names = ["Ground State"]
        for i in range(1, num_states):
            state_names += ["{}{} State".format(i, th.get(i, "th"))]

        # If we want to plot the potential, plot it first.
        all_psi = np.vstack(([V], all_psi))
        state_names = ["Potential"] + state_names

        # iterate over all the functions to plot, and plot them individually.
        for i in range(len(all_psi)):
            title = "The {} for the {} along $x$:".format(state_names[i], pot_sys_name)
            file_name = "state_{}".format(i - 1)
            state = "$\psi$"

            plt_with_V = include_V
            with_V_scale = V_scale
            if state_names[i] == "Potential":
                file_name = "potential"
                state = "V"
                plt_with_V = False
                with_V_scale = 1

            legend = [state_names[i]]
            if plt_with_V:
                legend = [state_names[0]] + legend
            legend = tuple(legend)

            plot_line(*r, all_psi[i], title, state, legend=legend, include_V=plt_with_V, V=V, V_scale=with_V_scale,
                      filename=file_name)

    # If the system is 2D, plot the img, wireframe and surfaces.
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

    # if the system is 3D, plot the 3D scatter.
    elif D == 3:

        if include_V:
            title = "The Potential function for the {} along $x$, $y$ & $z$".format(pot_sys_name)
            N = V.shape[1]
            D = len(V.shape)
            V = V.reshape(N ** D)
            plot_3D_scatter(*r, V, title)

        num_states = len(all_psi)
        for n in range(num_states):
            title = "$\psi_{}$ for the {} along $x$, $y$ & $z$".format(n, pot_sys_name)

            N = all_psi[n].shape[1]
            D = len(all_psi[n].shape)
            psi = all_psi[n].reshape(N ** D)

            plot_3D_scatter(*r, psi, title)

    # All higher order systems can't be easily visualised.
    else:
        return


def compute(start=-10, stop=10, N=100, D=1, num_states=1, num_iterations=10 ** 5):
    """
    The method to set up the variables and system, and aggregate the computed wavefunctions.
    :param start: The lower bound of the grid.
    :param stop: The upper bound of the grid.
    :param N: The number of samples along an axis.
    :param D: The number of dimensions.
    :param num_states: The number of wavefunctions to compute.
    :param num_iterations: The number of iterations per computation.
    :return: r, V, all_psi: the grid, potential function and the list of all the wavefunctions.
    """
    # Keep the number of states in bounds, so that the orthonormal basis generator doesn't return an error.
    if num_states >= N:
        num_states = N - 2

    # The coordinates along the x axis
    x = np.linspace(start, stop, N)
    # The axes along each dimension
    axes = [x] * D
    # populate the grid using the axes.
    r = np.array(np.meshgrid(*axes, indexing="ij"))

    # generate the potential for the system
    V = potential(r)

    # Calculate the grid spacing for the symmetric grid.
    dr = (stop - start) / N

    # Generate the 2nd order finite difference derivative matrix.
    generate_laplacian(D, N, dr)

    # Keep track whether we are on the first iteration or not.
    first_iteration = True
    # Set up two arrays to store the generated psi:
    # Stores the psi as linear column vectors, used for calculating the next psi in the series.
    all_psi_linear = np.zeros((1, N ** D))
    # stores in their proper shape as grids, used for plotting.
    all_psi = np.zeros((1, N * D))

    # iterate over the number of states we want to generate psi for.
    for i in range(num_states):
        # Generate the psi for this order number
        psi = nth_state(r, dr, D, N, num_iterations, all_psi_linear, i + 1)

        # Store the generated psi in both ways in their corresponding arrays.
        psi_linear = psi.reshape(N ** D)
        if first_iteration:
            all_psi_linear = np.array([psi_linear])
            all_psi = np.array([psi])
            first_iteration = False
        else:
            all_psi_linear = np.vstack((all_psi_linear, [psi_linear]))
            all_psi = np.vstack((all_psi, [psi]))

    return r, V, all_psi


def main():
    # Whether to plot the potential function or not.
    include_potential = True

    # The size and range of the grid
    start, stop, N = -10, 10, 100
    # The number of orders of psi to calculate
    num_states = 1
    # The number of axes of the system
    D = 1
    # Number of times to generate samples in the wavefunction
    num_iterations = 10 ** 5

    # a factor to scale the psi by when plotting it together with the potential function in the 1D case.
    v_scale = 10

    r, V, all_psi = compute(start, stop, N, D, num_states, num_iterations)
    # plot the generated psis.
    plotting(r, all_psi, D, include_potential, V, v_scale)


main()
