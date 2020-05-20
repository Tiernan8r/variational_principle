import random
import time

import numpy as np
from scipy.linalg import null_space

from plot.plot import plotting
from diffrentiation.laplacian import generate_laplacian
from integration.romberg import romberg
from potentials.potential import potential

# global constants:
hbar = 6.5821189 * 10 ** -16  # 6.582119569x10^-16 eV (from wikipedia)
# electron
m = 9.1093819 * 10 ** -31  # 9.1093837015(28)x10^-31
# factor used in calculation of energy
factor = -(hbar ** 2) / (2 * m)

# The Lagrangian Derivative matrix
global DEV2


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
    orthonormal_basis = null_space(prev_psi_linear).T

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
    global DEV2
    DEV2 = generate_laplacian(D, N, dr)

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
