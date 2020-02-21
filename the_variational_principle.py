import random

import matplotlib.pyplot as plt
import numpy

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


# def second_derivative(f: numpy.ndarray, i: int, dx: float):
#     inv_h_sq = dx ** -2
#     # forward difference for left edge:
#     if i == 0:
#         integrand = f[i] - 2 * f[i + 1] + f[i + 2]
#     # backward difference for right edge:
#     elif i == len(f) - 1:
#         integrand = f[i - 2] - 2 * f[i - 1] + f[i]
#     # central difference for non edge cases
#     else:
#         integrand = f[i - 1] - 2 * f[i] + f[i + 1]
#
#     return integrand * inv_h_sq

def second_derivative(f):
    global A
    return numpy.dot(A, f)


# def energy(psi: numpy.ndarray, V: numpy.ndarray, dx: float):
#     total_energy = 0
#     # for i in range(1, len(psi) - 1):
#     for i in range(len(psi)):
#
#         if not numpy.isnan(V[i]) or numpy.isinf(V[i]):
#             T = factor * second_derivative(psi, i, dx)
#             # T = factor * (psi[i - 1] - 2 * psi[i] + psi[i + 1]) * (dx ** -2)
#             Vp = V[i] * psi[i]
#             E = psi[i] * (T + Vp)
#
#             total_energy += E
#
#     return total_energy * dx


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


def ground_state(start, stop, num_points, num_iterations):
    random.seed("THE-VARIATIONAL-PRINCIPLE")

    dx = (stop - start) / num_points

    x = numpy.linspace(start, stop, num_points)
    V = potential(x)

    psi = numpy.ones(num_points)
    psi[0], psi[-1] = 0, 0

    psi = normalise(psi, dx)

    previous_energy = energy(psi, V, dx)
    print("Initial Energy:", previous_energy)

    for i in range(num_iterations):
        rand_x = random.randrange(1, num_points - 1)
        # rand_x = random.randrange(N)
        rand_y = random.random() * 0.1 * (num_iterations - i) / num_iterations

        if random.random() > 0.5:
            rand_y *= -1

        psi[rand_x] += rand_y
        psi = normalise(psi, dx)

        new_energy = energy(psi, V, dx)
        if new_energy < previous_energy:
            previous_energy = new_energy
        else:
            psi[rand_x] -= rand_y
            psi = normalise(psi, dx)

    print("Final Energy:", energy(psi, V, dx))

    return psi


def main():
    a, b, N, num_iterations = -10, 10, 100, 10 ** 5
    # a, b, N, num_iterations = 0, 10, 100, 10 ** 5
    x = numpy.linspace(a, b, N)
    V = potential(x)

    dx = (b - a) / N
    generate_derivative_matrix(N, dx)
    psi = ground_state(a, b, N, num_iterations)

    scale = 10
    plt.plot(x, V)
    plt.plot(x, psi * scale)

    plt.title("Ground State $\psi$ for the Perturbed Finite Square Well:")
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    # plt.legend(("Original $\psi$", "potential", "Normalised $\psi$", "Final $\psi$"))
    plt.legend(("Potential", "Ground State"))
    # plt.legend(("Ground State", "Analytical Solution"))
    plt.show()

    # #Comparison for actual inf sq well:
    # third = N // 3
    # psi_rel = psi[third:2 * third + 1]
    #
    # L = len(psi_rel) * dx
    # x_rel = numpy.linspace(0, L, len(psi_rel))
    # psi_anal = normalise(numpy.sin(numpy.pi / L * x_rel), dx)
    #
    # print(len(psi_rel))
    # plt.plot(x_rel, psi_rel)
    # plt.plot(x_rel, psi_anal)
    # plt.title("Comparison of Analytical to Computed $\psi$ for the Infinite Square Well:")
    # plt.xlabel("x")
    # plt.ylabel("$\psi$")
    # plt.legend(("Computed", "Analytical"))
    # plt.show()


main()
