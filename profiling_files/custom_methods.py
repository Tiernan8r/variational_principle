import random

import matplotlib.pyplot as plt
import numpy

a = -20.0
b = 20.0
n = 2 ** 10 + 1
# n = 10
h = (b - a) / n
inv_h_sq = h ** -2
half_h = h * 0.5

hbar = 1.054571817 * 10 ** -34
m = 9.1093837015 * 10 ** -31
factor = -hbar ** 2 / (2 * m)

x = numpy.linspace(a, b, n)
k = 0.01
# V = numpy.zeros(n)
V = 0.5 * k * x ** 2

global Is
global Es
global norms
global psi
global Hp
global E

Hp = numpy.zeros(n, dtype=complex)

Is = numpy.zeros(n, dtype=complex)
Es = numpy.zeros(n, dtype=complex)
is_normalised = False
norms = numpy.zeros(n, dtype=complex)
E = 0

psi = numpy.linspace(1, 1, n, dtype=complex)
mag_psi = psi.conj() * psi

# number_iterations = 10000
number_iterations = 50000


def re_integrate(i: int, f: numpy.ndarray, out: numpy.ndarray, step=h):
    # rectangular rule, at only the given index to change the array
    # sum the array after

    # f_i = Is[i]
    # Loop the indices back to the start if they overflow
    # f_i_plus_i = Is[(i + 1) % n]
    # Es[i] = (f_i + f_i_plus_i) * half_h

    length = len(f)
    f_i = f[i]
    f_i_plus_1 = f[(i + 1) % length]
    out[i] = (f_i + f_i_plus_1) * 0.5 * step


def second_derivative(f: numpy.ndarray, i: int, wrap=False):
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
    Hp[i] = Tpi + Vpi


def energy_expectation():
    # E = numpy.sum(Es).real
    # return E
    return numpy.sum(Es).real


def recalculate_energy(psi: numpy.ndarray, i: int):
    # psi*
    # H psi
    # I = psi* x H psi
    # get at index i
    # Change array entry at only this point,
    # Then sum the array

    psi_star = psi[i].conj()

    # print("psi:", psi[i])
    # print("psi*:",psi_star)
    # alters the H*psi value at index i
    # print("Hp0:",Hp[i])

    hamil(psi, i)

    # print("Hp'",Hp[i])
    # print("Is0",Is[i])

    Is[i] = psi_star * Hp[i]
    # print("Is'",Is[i])

    # re_integrate(i)
    re_integrate(i, Is, Es)


def re_norm(psi: numpy.ndarray, i: int):
    mag_psi[i] = psi[i].conj() * psi[i]
    re_integrate(i, mag_psi, norms)


def normalise(psi: numpy.ndarray):
    # Assumes that norms has been properly calculated...
    global is_normalised
    global norm
    global norms

    if not is_normalised:
        norm = numpy.sum(norms)
        psi /= numpy.sqrt(norm)
        norms /= norm
        is_normalised = True


def ground_state(psi_start: numpy.ndarray, number_iterations: int, seed="The Variational Principle"):
    global Es
    global Is
    global is_normalised

    # Set the default wavefunction
    psi = psi_start
    normalise(psi)
    # Get the number of entries in the psi array to generate the random index between.
    # number_entries = len(psi)
    E = energy_expectation()

    # Get the random number generator, uses the given seed so that the results are repeatable
    random.seed(seed)

    # Iterate for the number of desired iterations
    for i in range(number_iterations):

        # Get a random x coord to sample
        rand_x = random.randrange(0, n)

        # Generate a random number to alter the entry by.
        rand_tweak = random.random()

        # Get the current energy for reference.
        # E = energy_expectation(psi)

        # Tweak the value in psi upward
        # print("BEFORE:")
        # print(psi)
        # print("CHANGE: @", rand_x, "BY:", rand_tweak)
        psi[rand_x] += rand_tweak
        # print("AFTER:")
        # print(psi)
        # print()
        re_norm(psi, rand_x)
        normalise(psi)

        # E_up = energy_expectation(psi)
        recalculate_energy(psi, rand_x)
        E_up = energy_expectation()

        # Tweak the value in psi downward from the original psi value
        psi[rand_x] -= 2 * rand_tweak
        re_norm(psi, rand_x)
        normalise(psi)

        # E_down = energy_expectation(psi)
        recalculate_energy(psi, rand_x)
        E_down = energy_expectation()

        # reset psi to the original value
        psi[rand_x] += rand_tweak
        re_norm(psi, rand_x)
        normalise(psi)

        # print(i, ": E:", E, "Eu:", E_up, "Ed:", E_down)
        # Compare energies for tweaking the entry up versus down, and keep the change
        # that results in a lower overall expectation value for the energy.
        if E_up < E_down and E_up < E:
            # If increasing the value in the entry results in a lower overall <E>
            # set the change and keep it
            psi[rand_x] += rand_tweak
            E = E_up
            re_norm(psi, rand_x)
            is_normalised = False
            # print("CHOSE UP")
        elif E_down < E_up and E_down < E:
            # If decreasing the entry results in a lower overall <E>,
            # reduce the value and keep it.
            psi[rand_x] -= rand_tweak
            E = E_down
            re_norm(psi, rand_x)
            is_normalised = False
            # print("CHOSE DOWN")

        # otherwise the psi should be left unchanged.
        # Same goes for Is, Es, norms, and the normalisation.

    # Normalise the final wavefunction
    # psi = normalise_psi(psi)
    normalise(psi)
    return psi


def initialise():
    global E
    for i in range(n):
        recalculate_energy(psi, i)
        re_norm(psi, i)
    # mag_psi = numpy.sum(norms)
    # psi = psi / numpy.sqrt(mag_psi)
    normalise(psi)
    E = energy_expectation()
    print(E)


initialise()
# plt.plot(psi)
normalise(psi)
# plt.plot(psi)
# plt.show()
for i in range(n):
    re_norm(psi, i)

psi = ground_state(psi, number_iterations)
print(E)
# plt.plot(x, V)
plt.plot(x, psi)
plt.show()
