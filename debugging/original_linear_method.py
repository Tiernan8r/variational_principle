# Used for the generation of random numbers
import random

# Used for plotting
import matplotlib.pyplot as plt
# Used for numerical calculations throughout
import numpy
# Used for computing the laplacian in the Hamiltonian
import scipy.ndimage.filters as filters
# Used for Fourier Analysis of results
from scipy import fftpack
# Used for integration calculations
from scipy import integrate

h_bar = 1
# h_bar = 1.054571817 * 10 **-34 	#J⋅s
# h_bar = 6.582119569 * 10 ** -16   #eV.s


m = 1
# m = 9.1093837015 *10**-31 #kg (from wikipedia)

factor = -h_bar ** 2 / (2 * m)  # %% md

x_min = -20
x_max = -x_min
n = 7
number_samples = 2 ** n + 1  # for romberg integration condition, has to be
# a number equal to 2**n + 1
x_step = (x_max - x_min) / number_samples  # make the step accurate


def free_particle(r: numpy.ndarray):
    """
    The free particle potential represents an unbound particle in zero potential.
    :param r: The coordinates of the potential values
    :return: The free particle potential.
    """
    # The free particle has no potential.
    return numpy.zeros(len(r))


def finite_square_well(r: numpy.ndarray, V_0=1.0, wall_bounds=(200, 800)):
    """
    The infinite square well has defined potential within the bounds.
    :param r: The coordinates of the potential values.
    :param V_0: The potential value outside of the well
    :param wall_bounds: The points at which the well boundary starts.
    :return: The infinite potential well function.
    """
    # Get the number of points along the axis
    num_points = len(r)
    # The first number in wall_bounds is the index of the first well wall
    lower = wall_bounds[0]
    # the index of the second wall is the second index in wall_bounds
    upper = num_points - wall_bounds[1]

    # Create the potential barrier of the well.
    lower_wall = numpy.array([V_0] * lower)
    # inside the well the potential is 0
    zeroes = numpy.zeros(num_points - lower - upper)
    # second potential barrier
    upper_wall = numpy.array([V_0] * upper)
    # Make the potential the combination of all the sections
    pot = numpy.concatenate((lower_wall, zeroes, upper_wall))

    return pot


def infinite_square_well(r: numpy.ndarray, wall_bounds=(200, 800)):
    """
    The infinite square well has defined potential within the bounds.
    :param r: The coordinates of the potential values.
    :param wall_bounds: The points at which the well boundary starts.
    :return: The infinite potential well function.
    """
    # Use the finite square well with infinite barriers.
    return finite_square_well(r, numpy.inf, wall_bounds)


def harmonic_oscillator(r: numpy.ndarray, k=0.01):
    """
    The Harmonic Oscillator potential is quadratic about the origin.
    :param r: The coordinates of the potential values.
    :param k: Hooke's constant for the system
    :return: The harmonic potential.
    """
    # The potential energy formula:
    return 0.5 * k * r ** 2


def potential(r: numpy.ndarray):
    """
    The potential energy function of the system.
    :param r: The points on the axes of the system.
    :return: A scalar ndarray of the values of the potential at each point in r.
    """

    return harmonic_oscillator(r, 1)
    # return finite_square_well(r, 1, (200, 800))
    # return infinite_square_well(r, (200, 800))
    # return free_particle(r)


def gen_matrix(dimensions: int):
    mat = numpy.zeros((dimensions, dimensions))
    # populate with central difference:
    # ignore first and last rows
    for i in range(1, dimensions - 1):
        # for j in range(dimensions):
        #     if i == j:
        #         mat[i][j] = -2
        #     elif i - 1 == j or i +1 == j :
        #         mat[i][j] = 1
        mat[i][i - 1] = 1.0
        mat[i][i] = -2.0
        mat[i][i + 1] = 1.0

    # first entry is forward difference
    mat[0][0] = 1.0
    mat[0][1] = -2.0
    mat[0][2] = 1.0
    # Last entry is backward difference
    mat[-1][-1] = 1.0
    mat[-1][-2] = -2.0
    mat[-1][-3] = 1.0

    return mat


A = gen_matrix(number_samples)


def second_derivative(psi: numpy.ndarray):
    global A
    return numpy.dot(A, psi) * x_step ** -2


# V is the potential,
# psi is the wavefunction
def hamiltonian(psi: numpy.ndarray):
    """
        The Hamiltonian Operator H, operates on the given wavefunction psi according
        to the Schrodinger Time Indepedent Equation:
        H = -(hbar^2)/(2m) * d^2/dx^2 + V(x)
        :param psi: The wavefunction of the system.
        :return: The energy E of the wavefunction.
        """

    # The grad() function to take derivatives of the given wavefunction
    laplace = filters.laplace
    V = potential(x)

    # Since the Hamiltonian represents the Energy of the system, it can be split
    # into the potential energy (V) and kinetic energy (T)

    # The potential energy combined with the wavefunction.
    Vp = V * psi
    # The kinetic energy operated on the wavefunction, using the pre-calculated
    # factor of -hbar^2 / 2m.
    # Tp = factor * laplace(psi, mode = "nearest")
    Tp = factor * second_derivative(psi)
    return Tp + Vp


def normalise_psi(psi: numpy.ndarray):
    """
    Normalises the given wavefunction so that it's magnitude integrated over it's
    extent is 1.
    :param psi: The wavefunction to normalise
    :return: A normalised ndarray of the wavefunction psi.
    """

    # Get the complex conjugate of the wavefunction
    psi_star = psi.conj()
    # Get the magnitude of the wavefunction by multiply the original and conjugate
    # parts.
    mag_psi = psi_star * psi

    # Get the integration of this magnitude
    norm = integrate.romb(mag_psi)

    # Normalise the wavefunction by the square root of the norm.
    norm_psi = psi / numpy.sqrt(norm)

    return norm_psi


def expectation_value(Q, psi: numpy.ndarray):
    """
    Calculates the general expectation <Q> for the general operator Q.
    :param Q: Any operator on psi.
    :param psi: The wavefunction to operate on
    :return: The expectation <Q> of the operator.
    """
    # TODO: add check for if it's already normalised?
    # may be more efficient when large arrays are involved
    # psi = normalise_psi(psi)
    # Get the complex conjugate of the wavefunction
    psi_star = psi.conjugate()
    # The condition to integrate
    integrand = psi_star * Q(psi)

    norm = integrate.romb(psi_star * psi)

    return integrate.romb(integrand) / norm


def energy_expectation(psi: numpy.ndarray):
    """
    Calculates the expectation value for the energy of the given wavefunction
    representing a particle.
    :param psi: The wavefunction to get the energy of.
    :return: The <E> a scalar of the energy of the particle in the system.
    """
    # Get the potential of the system
    V = potential(x)
    # Get the Hamiltonian of the system
    H = hamiltonian
    # Calculate <H> which is the equivalent to <E>
    return expectation_value(H, psi)


x = numpy.linspace(x_min, x_max, number_samples)


# potential = finite_square_well
# potential = harmonic_oscillator

def ground_state(psi_start: numpy.ndarray, number_iterations: int, seed="The Variational Principle"):
    # Set the default wavefunction
    psi = psi_start
    # Get the number of entries in the psi array to generate the random index between.
    number_entries = len(psi)

    # Get the random number generator, uses the given seed so that the results are repeateable
    random.seed(seed)

    # Iterate for the number of desired iterations
    for i in range(number_iterations):

        # Get a random x coord to sample
        rand_x = random.randrange(0, number_entries)

        # Generate a random number to alter the entry by.
        rand_tweak = random.random() * (number_iterations - i) / number_iterations

        # TODO: remove this repeat calculation for optimisation
        # Get the current energy for reference.
        E = energy_expectation(psi)

        # Tweak the value in psi upward
        psi[rand_x] += rand_tweak
        E_up = energy_expectation(psi)

        # Tweak the value in psi downward from the original psi value
        psi[rand_x] -= 2 * rand_tweak
        E_down = energy_expectation(psi)

        # reset psi to the original value
        psi[rand_x] += rand_tweak

        # Compare energies for tweaking the entry up versus down, and keep the change
        # that results in a lower overall expectation value for the energy.
        if E_up < E_down and E_up < E:
            # If increasing the value in the entry results in a lower overall <E>
            # set the change and keep it
            psi[rand_x] += rand_tweak
        elif E_down < E_up and E_down < E:
            # If decreasing the entry results in a lower overall <E>,
            # reduce the value and keep it.
            psi[rand_x] -= rand_tweak
        # otherwise the psi should be left unchanged.

    # Normalise the final wavefunction
    psi = normalise_psi(psi)
    return psi


# The number of times to generate a random number.
# number_iterations = 50000
number_iterations = 10 ** 5

# Set the default wavefunction
psi = numpy.linspace(1, 1, number_samples)

# Plot the wavefunction versus the potential for visualisation
plt.plot(x, psi)
plt.plot(x, potential(x))
plt.legend(("$\psi$", "V"))
plt.show()

import time

t1 = time.time()
psi = ground_state(psi, number_iterations)
t2 = time.time()
print("Time for", number_iterations, "samples:", t2 - t1, "s.")

# plot the final wavefunction.
plt.plot(x, psi)
# plt.plot(x, potential(x))
plt.show()

energy = energy_expectation(psi)
print("Energy:", energy)

# Keep the wavefunction generated for reference.
saved_psi = psi

# Get the saved psi
psi = saved_psi

# Do the FT on the wavefunction
fft_psi = fftpack.fft(psi)
# Half the produced value as the FT is symmetric
fft_psi = fft_psi[:int(len(fft_psi) / 2)]
# plt.plot(x, fft_psi)

# Plot the FT to visualise the results.
plt.plot(fft_psi)
plt.show()

# Chop the FT to keep only the most important harmonics
fft_psi = fft_psi[:25]
# Plot the minimised FT
plt.plot(fft_psi)
plt.show()

# Perform an inverse FT to get the smoothed wavefunction back
psi = fftpack.ifft(fft_psi)

# Normalise the result
psi_conj = psi.conj()
mod_psi = psi_conj * psi
A = numpy.sqrt(integrate.simps(mod_psi))
psi /= A

# Plot the final wavefunction to show the result.
x_range = numpy.linspace(x_min, x_max, len(psi))
plt.plot(x_range, psi)
plt.title("Smoothed Wavefunction $\psi$")
plt.xlabel("x")
plt.ylabel("$\psi$")
plt.show()
