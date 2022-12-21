import numpy as np

# def composite_trapezoidal(f, a, b, j):
#     h = b - a
#
#     f_a = f(a)
#     f_b = f(b)
#
#     i = np.linspace(1, 2**(j-1)-1)
#     f_j = 0
#     return


def romberg(f, a, b, eps, J = 10):
    h = b - a
    T = []
    # fa, fb = f[a], f[b]
    for j in range(1, J):
        # tot = 0
        # for i in range(1, 2**(j + 1) - 1):
        #     tot += f[a + j * h]
        # T[j,1] = h / 2 * (fa + 2 * tot + fb)
        tot = 0
        for i in range(1, 2**(j -2)):
            tot += f[a + (2j - 1) * h]

        prev_T = 0
        if j - 1 > 1:
            prev_T = T[j - 1][1]

        T += [[0.5 * prev_T + h * tot]]

        for k in range(2, j):
            T[j] += [T[j][k - 1] + (T[j][k-1] - T[j-1][k-1]) / (4**(k-1) -1)]

        acc = T[j - 1][-1] - T[j][-1]
        if acc < eps:
            return T[j][-1], acc

        h /= 2

    return T[j][-1], acc



