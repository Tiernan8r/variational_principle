import random
import time

import matplotlib.pyplot as plt
import numpy
import scipy.linalg as linalg

# global constants:
hbar = 6.582119569 * 10 ** -16  # 6.582119569×10−16 (from wikipedia)
m = 9.1093837015 * 10 ** -31  # 9.1093837015(28)×10−31
factor = -(hbar ** 2) / (2 * m)


def normalise(psi: numpy.ndarray, dx: float):
    # integrate using the rectangular rule
    norm = numpy.sum(psi * psi) * dx
    norm_psi = psi / numpy.sqrt(norm)
    return norm_psi


global A


def generate_derivative_matrix(dimensions: int, dx):
    global A
    A = numpy.zeros((dimensions, dimensions))
    for i in range(1, dimensions - 1):
        A[i][i - 1], A[i][i], A[i][i + 1] = 1, -2, 1
    A[0][0], A[-1][-1], A[0][2], A[-1][-3] = 1, 1, 1, 1
    A[0][1], A[-1][-2] = -2, -2
    return A * (dx ** -2)


def second_derivative(f):
    global A
    # return numpy.dot(A, f)
    return A @ f


def energy(psi: numpy.ndarray, V: numpy.ndarray, dx: float):
    Vp = V * psi
    Tp = factor * second_derivative(psi)
    return numpy.sum(psi * (Tp + Vp)) * dx


def potential(x: numpy.ndarray):
    # return 0.5 * x ** 2

    # length = len(x)
    # third = length // 3
    # # mid, bef = numpy.zeros(third + 1), numpy.linspace(numpy.inf, numpy.inf, third)
    # mid, bef = numpy.zeros(third + 1) + 0.3 * numpy.linspace(0, (x[1] - x[0]) * (third + 1), third + 1), numpy.linspace(
    #     10, 10, third)
    # aft = bef.copy()
    # return numpy.concatenate((bef, mid, aft))

    return 0.5 * x ** 2


def generate_orthogonal_states(pre_existing_states: numpy.ndarray, size):
    # there are no known states already
    if pre_existing_states.size == 0:
        return numpy.identity(size)
    else:
        orthogonal_states = linalg.null_space(pre_existing_states)
        return orthogonal_states.transpose()


def nth_state(start: float, stop: float, dimension: int, num_iterations: int, previous_states: numpy.ndarray):
    # the iteration number
    n = 0
    if previous_states.size != 0:
        n = previous_states.shape[0]

    t1 = time.time()
    states = generate_orthogonal_states(previous_states, dimension)
    # Get the number of rows
    row_size = states.shape[0]

    random.seed("THE-VARIATIONAL-PRINCIPLE")

    dx = (stop - start) / dimension

    x = numpy.linspace(start, stop, dimension)
    V = potential(x)

    psi = numpy.ones(dimension)
    psi[0], psi[-1] = 0, 0

    psi = normalise(psi, dx)

    previous_energy = energy(psi, V, dx)
    print("Initial Energy:", previous_energy)

    for i in range(num_iterations):
        # rand_x = random.randrange(1, dimension - 1)
        rand_x = random.randrange(1, row_size - 1)
        rand_y = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_y *= -1

        psi += states[rand_x] * rand_y
        psi = normalise(psi, dx)

        new_energy = energy(psi, V, dx)
        if new_energy < previous_energy:
            previous_energy = new_energy
        else:
            psi -= states[rand_x] * rand_y
            psi = normalise(psi, dx)

    print("Final Energy:", energy(psi, V, dx))
    t2 = time.time()
    print("The time for the " + str(n) + "th iteration is:", t2 - t1, "s.\n")

    plt.plot(x, psi)
    plt.title("The {}th State for the Harmonic Oscillator:".format(n))
    plt.ylabel("$\psi$")
    plt.xlabel("x")
    plt.show()

    return psi


def main():
    a, b, N, num_iterations = -10, 10, 100, 10 ** 5
    # a, b, N, num_iterations = -10, 10, 1000, 10 ** 6
    x = numpy.linspace(a, b, N)
    # V = potential(x)

    dx = (b - a) / N
    generate_derivative_matrix(N, dx)
    existing_states = numpy.array([])
    number_states = 1
    for i in range(number_states):
        psi = nth_state(a, b, N, num_iterations, existing_states)
        # existing_states += [psi]
        if existing_states.size == 0:
            existing_states = numpy.array([psi])
        else:
            existing_states = numpy.vstack((existing_states, psi))

    for j in range(existing_states.shape[0]):
        plt.plot(x, existing_states[j])

    plt.title("Wavefunctions $\psi$ for the Linear Harmonic Oscillator:")
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    # # plt.legend(("Original $\psi$", "potential", "Normalised $\psi$", "Final $\psi$"))
    # plt.legend(("Potential", "Ground State", "Second State", "Third State", "Fourth State", "..."))
    plt.legend(("Ground State", "Second State", "Third State", "Fourth State", "..."))
    # # plt.legend(("Ground State", "Analytical Solution"))
    plt.show()

    ground_psi = existing_states[0]
    orthogonal_states = generate_orthogonal_states(existing_states, N)
    delta = 0.01
    for j in range(len(orthogonal_states)):
        if abs(orthogonal_states[j][0]) > delta:
            plt.plot(x, orthogonal_states[j])
            print(orthogonal_states[j][0])
    plt.title("Error in Orthogonal States:")
    plt.show()


main()
