import random

import matplotlib.pyplot as plt
import numpy
# Used for Fourier Analysis of results
from scipy import fftpack

# a = -20.0
a = 0
b = 20.0
n = 2 ** 10 + 1
# n = 10
h = (b - a) / n
inv_h_sq = h ** -2
half_h = h * 0.5

hbar = 1.054571817 * 10 ** -34  # eV
m = 9.1093837015 * 10 ** -31  # kg
factor = -hbar ** 2 / (2 * m)

x = numpy.linspace(a, b, num=n, dtype=complex)
k = 0.01
# V = numpy.zeros(n)
V = 0.5 * k * x ** 2

# Array containing all the H(psi)
global hamiltonians_array
# The tiny psi x H(psi)
global infinitesimal_energy_expectations
# The actual [psi x H(psi)]dx to sum over
global energies_array
# Array of magnitudes of the psi
global normalisation_array
# psi itself
global psi
# The actual <E> as a sum of the tiny <E>
global E

hamiltonians_array = numpy.zeros(n, dtype=complex)

infinitesimal_energy_expectations = numpy.zeros(n, dtype=complex)
energies_array = numpy.zeros(n, dtype=complex)
is_normalised = False
normalisation_array = numpy.zeros(n, dtype=complex)
E = 0

psi = numpy.linspace(1, 1, n, dtype=complex)
mag_psi = psi.conj() * psi

# number_iterations = 10000
number_iterations = 50000


def re_integrate(i: int, f: numpy.ndarray, step=h):
    # rectangular rule, at only the given index to change the array
    # sum the array after

    # f_i = Is[i]
    # Loop the indices back to the start if they overflow
    # f_i_plus_i = Is[(i + 1) % n]
    # Es[i] = (f_i + f_i_plus_i) * half_h

    # Central Difference Rule:
    # length = len(f)
    # f_i = f[i]
    # f_i_plus_1 = f[(i + 1) % length]
    # out[i] = (f_i + f_i_plus_1) * 0.5 * step

    # Rectangular Rule:
    return f[i] * step


def second_derivative(f: numpy.ndarray, i: int, wrap=False):
    """
    Calculates the 2nd order finite difference on the array f at the index i. If wrap is False,
    The edge cases are calculated using the forward and backward differences respectively, otherwise the central
    difference method is used throughout.
    :param f: The array to perform the differentiation on.
    :param i: The index to perform the differentiation at.
    :param wrap: Whether the edge cases wrap around or not.
    :return: The second order derivative at the index i.
    """
    # https: // en.wikipedia.org / wiki / Finite_difference
    # h = step size
    # Centre difference 2nd Derivative: d^2f / dx^2 = f''(x):
    # 1/h**2 * [f_i+1 - 2f_i + f_i-1]
    length = len(f)

    if not wrap and i + 1 >= length:
        # Use second order backward instead:
        # f'' = 1/h**2 * [f_i + f_i-2 - 2f_i-1]
        f_i = f[i]
        f_i_minus_1 = f[i - 1]
        f_i_minus_2 = f[i - 2]

        div2 = inv_h_sq * (f_i - 2 * f_i_minus_1 + f_i_minus_2)
        return div2

    if not wrap and i - 1 < 0:
        # Use second order forward instead:
        # f'' = 1/h**2 * [f_i + f_i+2 - 2f_i+1]
        f_i = f[i]
        f_i_plus_1 = f[i + 1]
        f_i_plus_2 = f[i + 2]

        div2 = inv_h_sq * (f_i - 2 * f_i_plus_1 + f_i_plus_2)
        return div2

    # Prevent the index going out of range, by looping back the index if it's too high
    f_i_plus_1 = f[(i + 1) % length]
    # Assume that the current index is in range
    f_i = f[i]
    # The lower index inherently loops back using negative indexing
    f_i_minus_1 = f[i - 1]

    # Current derivative.
    div_2 = inv_h_sq * (f_i_plus_1 - 2 * f_i + f_i_minus_1)
    return div_2


def hamil(psi: numpy.ndarray, i: int):
    """
    Calculates the Hamiltonian of the given psi wavefunction at the given index in the array.
    :param psi: The wavefunction to operate on.
    :param i: The index to do the operation at.
    :return: The hamiltonian value at index i for the given psi.
    """
    # Tp = factor * second_derivative(psi)
    # Vp = V * psi
    # Hp = Tp + Vp
    # return Hp

    # Hp[i] = Tpi + Vpi
    # pass

    Tpi = factor * second_derivative(psi, i)
    Vpi = V[i] * psi[i]
    # Hpi = Tpi + Vpi
    # return Hpi
    # hamiltonians_array[i] = Tpi + Vpi
    return Tpi + Vpi


def recalculate_energy(psi: numpy.ndarray, i: int):
    """
    Recalculates the <E> at the given i for the given psi.
    :param psi: The wavefunction to use for the expectation value.
    :param i: The index to perform the calculation at.
    :return:
    """
    # psi*
    # H psi
    # I = psi* x H psi
    # get at index i
    # Change array entry at only this point,
    # Then sum the array
    global hamiltonians_array
    global energies_array
    global infinitesimal_energy_expectations

    # alters the H*psi value at index i
    hamiltonians_array[i] = hamil(psi, i)

    infinitesimal_energy_expectations[i] = psi[i].conj() * hamiltonians_array[i]

    # re_integrate(i)
    energies_array[i] = re_integrate(i, infinitesimal_energy_expectations)


def energy_expectation():
    """
    Calculates the <E> energy expectation value from the energues_array array
    :return: The energy expectation value <E> as a scalar.
    """
    # global E
    # E = numpy.sum(energies_array).real
    # return E
    return numpy.sum(energies_array).real


def re_norm(psi: numpy.ndarray, i: int):
    """
    Recalculates the normalisation for the given psi wavefunction at the index i.
    :param psi: The wavefunction to re normalise.
    :param i: The index to perform the normalisation at.
    :return:
    """
    global mag_psi
    global normalisation_array
    mag_psi[i] = psi[i].conj() * psi[i]
    normalisation_array[i] = re_integrate(i, mag_psi)
    # return re_integrate(i, mag_psi)


def normalise(psi: numpy.ndarray):
    """
    Re-normalise the wavefunction psi, using the normalisation factors pre calculated from normalisation_array.
    :param psi: The wavefunction to normalise.
    :return: The normalised wavefunction.
    """
    # Assumes that norms has been properly calculated...
    global is_normalised
    global norm
    global normalisation_array

    # if not is_normalised:
    #     norm = numpy.sum(norms)
    #     psi /= numpy.sqrt(norm)
    #     norms /= norm
    #     is_normalised = True

    norm = numpy.sum(normalisation_array)
    psi /= numpy.sqrt(norm)
    normalisation_array /= norm
    return psi
    # is_normalised = True


def ground_state(psi_start: numpy.ndarray, number_iterations: int, seed="The Variational Principle"):
    """
    Calculates the ground state wavefunction for a given potential system, by fiinding the wavefunction with
    minimum expectation value in the energy <E>.
    :param psi_start: The initial wavefunction guess.
    :param number_iterations: Number of times the wavefunction should be modified to obtain the final result.
    :param seed: A seed for the random number generator.
    :return: The normalised gorund state wavefunction.
    """
    global energies_array
    global infinitesimal_energy_expectations
    global is_normalised
    global E

    # Set the default wavefunction
    psi = psi_start
    # psi is already normalised by intialise()
    # and E is already calculated.

    # Get the random number generator, uses the given seed so that the results are repeatable
    random.seed(seed)

    # Iterate for the number of desired iterations
    for i in range(number_iterations):

        # Get a random x coord to sample
        rand_x = random.randrange(0, n)

        # Generate a random number to alter the entry by.
        rand_y = random.random()

        # Tweak the value in psi upward
        psi[rand_x] += rand_y

        # Recompute the normalisation for this entry
        re_norm(psi, rand_x)
        # Re normalise psi
        psi = normalise(psi)

        # Re calculate the energy at this entry as well
        recalculate_energy(psi, rand_x)
        # The tweaked up value is the new <E>
        E_up = energy_expectation()

        # Tweak the value in psi downward from the original psi value
        psi[rand_x] -= 2 * rand_y
        # Re normalise
        re_norm(psi, rand_x)
        psi = normalise(psi)

        # Calculate <E> for this change again.
        recalculate_energy(psi, rand_x)
        E_down = energy_expectation()

        # reset psi to the original value
        psi[rand_x] += rand_y
        re_norm(psi, rand_x)
        psi = normalise(psi)

        # Compare energies for tweaking the entry up versus down, and keep the change
        # that results in a lower overall expectation value for the energy.
        if E_up < E_down and E_up < E:
            # If increasing the value in the entry results in a lower overall <E>
            # set the change and keep it
            psi[rand_x] += rand_y
            E = E_up
            re_norm(psi, rand_x)
            psi = normalise(psi)
            # is_normalised = False
            # print("CHOSE UP")
        elif E_down < E_up and E_down < E:
            # If decreasing the entry results in a lower overall <E>,
            # reduce the value and keep it.
            psi[rand_x] -= rand_y
            E = E_down
            re_norm(psi, rand_x)
            psi = normalise(psi)
            # is_normalised = False
            # print("CHOSE DOWN")

        # otherwise the psi should be left unchanged.
        # Same goes for Is, Es, norms, and the normalisation.

    # Normalise the final wavefunction
    # psi = normalise_psi(psi)
    psi = normalise(psi)
    return psi


# p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }} e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },
def generate_psi():
    """
    Creates a Gaussian distribution for the wavefunction psi
    :return:
    """
    global psi
    mu = complex((a + b) / 2.0)
    sigma = complex((b - a) / 4.0)
    pi = complex(numpy.pi)
    A = 1 / numpy.sqrt(2 * pi * sigma ** 2)
    B = numpy.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))
    psi = A * B


def initialise():
    """
    Sets the initial values for the global arrays.
    :return:
    """
    global E
    global psi

    for i in range(n):
        re_norm(psi, i)
    psi = normalise(psi)

    for i in range(n):
        recalculate_energy(psi, i)

    # mag_psi = numpy.sum(norms)
    # psi = psi / numpy.sqrt(mag_psi)
    E = energy_expectation()


generate_psi()

initialise()
for i in range(n):
    re_norm(psi, i)
normalise(psi)


def plurts():
    plt.plot(x, V)
    plt.title("Potential")
    plt.show()

    plt.plot(x, psi)
    plt.title("$\psi$")
    plt.show()

    plt.plot(infinitesimal_energy_expectations)
    plt.title("Infinitesimal <E>s")
    plt.show()

    plt.plot(energies_array)
    plt.title("Infinitesimal Energies")
    plt.show()

    plt.plot(mag_psi)
    plt.title("$|\psi|^2$")
    plt.show()

    plt.plot(normalisation_array)
    plt.title("NORMS")
    plt.show()

    plt.plot(hamiltonians_array)
    plt.title("Infinitesimal Hamiltonians:")
    plt.show()


print("Initial Energy:", E)
# plurts()
psi = ground_state(psi, number_iterations)
print("Final Energy:", E)
# plurts()

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

# Plot the final wavefunction to show the result.
x_range = numpy.linspace(a, b, len(psi))
# plt.plot(x_range, psi)
# plt.title("Smoothed Wavefunction $\psi$")
# plt.xlabel("x")
# plt.ylabel("$\psi$")
# plt.show()
