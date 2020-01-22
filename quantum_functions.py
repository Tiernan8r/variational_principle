import numpy
import scipy.integrate as integrate

accuracy = 10 ** -6

h_bar = 1
# h_bar = 6.62607015* 10**-34 / (2 * numpy.pi)
# h_bar = 6.62607015* 10**-3 / (2 * numpy.pi) # from wikipedia (less 10**-31)
m = 1
# m = 9.1093837015 *10**-31 #kg (from wikipedia)
# m = 9.1093837015 #kg (from wikipedia) (less 10**-31)
factor = -h_bar ** 2 / (2 * m)

n = 3
number_samples = 2 ** n + 1  # for romberg integration

# x_min = -10
x_min = 0
# x_max = -x_min
x_max = 10
x_step = (x_max - x_min) / number_samples  # make the step accurate

x = numpy.linspace(x_min, x_max, number_samples)


def potential(x: numpy.ndarray):
    # use x to iterate if need be
    # per_section = int(numpy.floor(number_samples / 3))
    # excess = number_samples - per_section * 3
    # inf_wall = numpy.linspace(numpy.inf, numpy.inf, per_section)
    # zero_well = numpy.zeros(per_section + excess)
    # well = numpy.concatenate((-inf_wall, zero_well, inf_wall), axis=None)
    #
    # return well
    return numpy.zeros(number_samples)


# V is the potential,
# psi is the wavefunction
def hamiltonian(V: numpy.ndarray):
    grad = numpy.gradient

    def foo(psi: numpy.ndarray):
        Vp = V * psi  # The potential energy
        Tp = factor * grad(grad(psi))  # The kinetic energy
        return Tp + Vp

    return foo


def energy_expectation(psi: numpy.ndarray):
    V = potential(x)
    H = hamiltonian(V)
    psi_star = psi.conjugate()
    integrand = psi_star * H(psi)

    initial_integrand = integrate.romb(integrand, dx=x_step)

    # Normalisation
    abs_psi = numpy.abs(psi_star * psi)
    magnitude_psi = integrate.romb(abs_psi, dx=x_step)

    normalised = initial_integrand
    if not (magnitude_psi - accuracy) < 0:
        normalised /= magnitude_psi

    return normalised
