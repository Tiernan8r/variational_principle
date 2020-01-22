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

n = 2
number_samples = 2 ** n + 1  # for romberg integration

# x_min = -10
x_min = 0
# x_max = -x_min
x_max = 10
x_step = (x_max - x_min) / number_samples  # make the step accurate

x = numpy.linspace(x_min, x_max, number_samples)


def potential(x: numpy.ndarray):
    # use x to iterate if need be
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


def normalise_psi(psi: numpy.ndarray):
    psi_star = psi.conj()
    mag_psi = psi_star * psi

    norm = integrate.romb(mag_psi)
    norm_psi = psi / numpy.sqrt(norm)

    return norm_psi


def is_psi_normalised(psi: numpy.ndarray):
    psi_star = psi.conj()
    mag_psi = psi_star * psi

    norm = integrate.romb(mag_psi)

    return (1 - norm) > accuracy


def energy_expectation(psi: numpy.ndarray):
    V = potential(x)
    H = hamiltonian(V)
    return expectation_value(H, psi)


def expectation_value(Q, psi: numpy.ndarray):
    if not is_psi_normalised(psi):
        psi = normalise_psi(psi)
    psi_star = psi.conjugate()
    integrand = psi_star * Q(psi)

    return integrate.romb(integrand)
