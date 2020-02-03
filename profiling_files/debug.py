# The Variational Principle

import importlib
# Used for the generation of random numbers
import random

# Used for plotting
import matplotlib.pyplot as plt
import numpy
import scipy.integrate as integrate
# V is the potential,
# psi is the wavefunction
import scipy.ndimage.filters as filters
# Used for numerical calculations throughout
# Used for computing the laplacian in the Hamiltonian
# Used for Fourier Analysis of results
from scipy import fftpack

# h_bar = 1
# h_bar = 1.054571817 * 10 **-34 	#J⋅s
h_bar = 6.582119569 * 10 ** -16  # eV.s

# m = 1
m = 9.1093837015 * 10 ** -31  # kg (from wikipedia)

factor = -h_bar ** 2 / (2 * m)


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

    return harmonic_oscillator(r, 0.01)
    # return finite_square_well(r, 1, (200, 800))
    # return infinite_square_well(r, (200, 800))
    # return free_particle(r)


def hamiltonian(V: numpy.ndarray):
    """
    Returns the Hamiltonian Operator H for a given potential function V.

    :param V: The potential function of the system.
    :return: The Hamiltonian operator H of the system.
    """

    # The grad() function to take derivatives of the given wavefunction
    grad = numpy.gradient
    laplace = filters.laplace

    # In order to make the hamiltonian an operator, need to return a function
    # that can be the operator

    # placeholder function to be returned, takes in the wavefunction as it's parameter
    def foo(psi: numpy.ndarray):
        """
        The Hamiltonian Operator H, operates on the given wavefunction psi according
        to the Schrodinger Time Indepedent Equation:
        H = -(hbar^2)/(2m) * d^2/dx^2 + V(x)
        :param psi: The wavefunction of the system.
        :return: The energy E of the wavefunction.
        """
        # Since the Hamiltonian represents the Energy of the system, it can be split
        # into the potential energy (V) and kinetic energy (T)

        # The potential energy combined with the wavefunction.
        Vp = V * psi
        # The kinetic energy operated on the waefunction, using the pre-calculated
        # factor of -hbar^2 / 2m.
        # Tp = factor * grad(grad(psi))
        # Tp = factor * numpy.sum(grad(grad(psi)))
        Tp = factor * laplace(psi, mode="nearest")
        # H = Tp + Vp
        return Tp + Vp

    # return the operator function
    return foo


def normalise_psi(psi: numpy.ndarray):
    """
    Normalises the given wavefunction so that it's magnitude integrated over it's
    extent is 1.
    :param psi: The wavefunction to normalise
    :return: A normalised ndarray of the wavefunction psi.
    """
    # Can't use:
    # norm = expectation_value(lambda p: p, psi)
    # Due to recursive loop, solution is to make this function
    # stand alone.

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
    psi = normalise_psi(psi)
    # Get the complex conjugate of the wavefunction
    psi_star = psi.conjugate()
    # The condition to integrate
    integrand = psi_star * Q(psi)

    return integrate.romb(integrand)


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
    H = hamiltonian(V)
    # Calculate <H> which is the equivalent to <E>
    return expectation_value(H, psi)


def ground_state(psi_start: numpy.ndarray, number_iterations: int, seed="The Variational Principle"):
    # Set the default wavefunction
    psi = psi_start
    # set the E to start with
    E = energy_expectation(psi)
    # Get the number of entries in the psi array to generate the random index between.
    number_entries = len(psi)

    # Get the random number generator, uses the given seed so that the results are repeateable
    rand = random.seed(seed)

    # Iterate for the number of desired iterations
    for i in range(number_iterations):

        # Get a random x coord to sample
        rand_x = random.randrange(0, number_entries)

        # Generate a random number to alter the entry by.
        rand_tweak = random.random()

        # Get the current energy for reference.
        # E = energy_expectation(psi)

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
            psi[rand_x] += 2 * rand_tweak
            E = E_up
        elif E_down < E_up and E_down < E:
            # If decreasing the entry results in a lower overall <E>,
            # reduce the value and keep it.
            psi[rand_x] -= 2 * rand_tweak
            E = E_down
        # otherwise the psi should be left unchanged.

    # Normalise the final wavefunction
    psi = normalise_psi(psi)
    return psi


x_min = -20
x_max = -x_min
n = 10
number_samples = 2 ** n + 1  # for romberg integration
x_step = (x_max - x_min) / number_samples  # make the step accurate

x = numpy.linspace(x_min, x_max, number_samples)

# potential = finite_square_well
potential = finite_square_well

# The number of times to generate a random number.
# number_iterations = 5000
number_iterations = 1000
# number_iterations = 10
# number_iterations = 50000
# number_iterations = 5000000

# Set the default wavefunction
psi = numpy.linspace(0.2, 0.2, number_samples)

# Plot the wavefunction versus the potential for visualisation
# plt.plot(x, psi)
# plt.plot(x, potential(x))
# plt.legend(("$\psi$", "V"))
# plt.show()

psi = ground_state(psi, number_iterations)

# plot the final wavefunction.
# plt.plot(x, psi)
# plt.plot(x, potential(x))
# plt.show()

# import time
#
# nums = [5, 50, 500, 5000, 50000]
# times = []
# for number_iterations in nums:
#     psi = numpy.linspace(0.2, 0.2, number_samples)
#     t1 = time.time()
#     psi = ground_state(psi, number_iterations)
#     t2 = time.time()
#     time_elapsed = t2 - t1
#     times += [time_elapsed]
#
#     print("N:", number_iterations)
#     print("dt:", time_elapsed)
#
#     plt.plot(x, psi)
#     plt.show()
#
# plt.plot(nums, times)
# plt.show()


# Do the FT on the wavefunction
fft_psi = fftpack.fft(psi)
# Half the produced value as the FT is symmetric
fft_psi = fft_psi[:int(len(fft_psi) / 2)]
# plt.plot(x, fft_psi)

# Plot the FT to visualise the results.
# plt.plot(fft_psi)
# plt.show()

# Chop the FT to keep only the most important harmonics
fft_psi = fft_psi[:25]
# Plot the minimised FT
# plt.plot(fft_psi)
# plt.show()

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
