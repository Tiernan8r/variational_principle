import numpy as np
import scipy.sparse as spr

N = 3
num = 3

B = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
              [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])


# A = np.array([[["1.1.1","2.1.1","3.1.1"], ["1.2.1","2.2.1","3.2.1"], ["1.3.1","2.3.1","3.3.1"]], [["1.1.2","2.1.2","3.1.2"], ["1.2.2","2.2.2","3.2.2"], ["1.3.2","2.3.2","3.3.2"]], [["1.1.3","2.1.3","3.1.3"], ["1.2.3","2.2.3","3.2.3"], ["1.3.3","2.3.3","3.3.3"]]])
# A = np.array([[["1.1.1","1.2.1","1.3.1"],["2.1.1","2.2.1","2.3.1"],["3.1.1","3.2.1","3.3.1"]],[["1.1.2","1.2.2","1.3.2"],["2.1.2","2.2.2","2.3.2"],["3.1.2","3.2.2","3.3.2"]],[["1.1.3","1.2.3","1.3.3"],["2.1.3","2.2.3","2.3.3"],["3.1.3","3.2.3","3.3.3"]]])
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# A = np.ones([N] * num)
# print("A")
# print(A)
#
# U = A.reshape(N ** num)
# print("U")
# print(U)

# Q = U.reshape([N] * num)
# print()
# print(Q)


# # X DERIVATIVE: needs 1/dx^2 scalar
# diags_x = [[-2] * N ** num,
#            ([1] * (N - 1) + [0]) * (N**(num - 1)),
#            ([1] * (N - 1) + [0]) * (N**(num - 1))]
# # print(diags_x)
# D_x = spr.diags(diags_x, [0, -1, 1], shape=(N ** num, N ** num))
# print("D_x")
# print(D_x.toarray())


# ## Y DERIVATIVE: needs 1/dy^2 scalar
# diags_y = [[-2] * N ** num,
#            (([1] * N**(num-1)) * (N - 1) + [0] * N) * N**(num - 2),
#            (([1] * N**(num-1)) * (N - 1) + [0] * N) * N**(num - 2)]
# # tmp = (([1] * N) * (N - 1) + [0] * N)
# # print(tmp)
# # print(diags_y)
# D_y = spr.diags(diags_y, [0, -N, N], shape=(N ** num, N ** num))
# print("D_y")
# print(D_y.toarray())

# ## Z DERIVATIVE: needs 1/dz^2 scalar:
# ...

# U_2x = D_x @ U
# U_2y = D_y @ U
#
# A_2x = U_2x.reshape([N] * num)
# A_2y = U_2y.reshape([N] * num)
#
# print("A_2x")
# print(A_2x)
# print("A_2y")
# print(A_2y)
#
# A_2 = A_2x + A_2y
# print("A_2")
# print(A_2)
#
# U_B_2 = (D_x + D_y) @ U
# B_2 = U_B_2.reshape([N] * num)
# print("B_2")
# print(B_2)


# def general_D(num: int, N: int, axis: str):
#     # make axis into an integer.
#
#     if axis == "x":
#         # axis# = 1... => for num = 3, num - 3 = 0 == axis#
#         axis_number = 0
#         # axis_number = num - 3
#         # X DERIVATIVE: needs 1/dx^2 scalar
#         diags_x = [[-2] * N ** num,
#                    ([1] * N ** axis_number * (N - 1) + [0] * N ** axis_number) * N**(num - 1),
#                    ([1] * N ** axis_number * (N - 1) + [0] * N ** axis_number) * N**(num - 1)]
#         print(diags_x)
#         D_x = spr.diags(diags_x, [0, -1, 1], shape=(N ** num, N ** num))
#         print("D_x")
#         print(D_x.toarray())
#         return D_x
#     elif axis == "y":
#         # axis# = 1... => for num = 3, num - 2 = 1 == axis#
#         axis_number = 1
#         # axis_number = num - 2
#         ## Y DERIVATIVE: needs 1/dy^2 scalar
#         diags_y = [[-2] * N ** num,
#                    ([1] * N**axis_number * (N - 1) + [0] * N**axis_number) * N ** (num - 2),
#                    ([1] * N**axis_number * (N - 1) + [0] * N**axis_number) * N ** (num - 2)]
#         print(diags_y)
#         D_y = spr.diags(diags_y, [0, -N, N], shape=(N ** num, N ** num))
#         print("D_y")
#         print(D_y.toarray())
#         return D_y
#     elif axis == "z":
#         # axis# = 2... => for num = 3, num - 1 = 2 == axis#
#         axis_number = 2
#         # axis_number = num - 1
#         ## Z DERIVATIVE: needs 1/dz^2 scalar
#         diags_z = [[-2] * N ** num,
#                    (([1] * N**axis_number) * (N - 1) + [0] * N**axis_number) * N ** (num - 3),
#                    (([1] * N**axis_number) * (N - 1) + [0] * N**axis_number) * N ** (num - 3)]
#         print(diags_z)
#         D_z = spr.diags(diags_z, [0, -N**2, N**2], shape=(N ** num, N ** num))
#         print("D_z")
#         print(D_z.toarray())
#         return D_z
#     else:
#         return 0

def general_D(num_axes: int, N: int, axis_number: int):
    # missing the 1/dr^2 factor

    # cap axis_number in range to prevent errors.
    axis_number %= num_axes

    num_repeats = num_axes - (axis_number + 1)
    diags = [[-2] * N ** num_axes,
             (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_repeats,
             (([1] * N ** axis_number) * (N - 1) + [0] * N ** axis_number) * N ** num_repeats]
    # print(diags)
    D = spr.diags(diags, [0, -N ** axis_number, N ** axis_number], shape=(N ** num_axes, N ** num_axes))
    # print("D_{}".format(axis_number))
    # print(D.toarray())
    return D


def generate_derivative_matrix(num_axes: int, axis_length: int, dr: float):
    global DEV2

    DEV2 = None

    for ax in range(num_axes):
        D = general_D(num_axes, axis_length, ax)
        if DEV2 is None:
            DEV2 = D
        else:
            DEV2 += D

    DEV2 *= (dr ** -2)


# D = general_D(num, N, 2)
# print()
# print(D)

generate_derivative_matrix(num, N, 1)
# print(DEV2.toarray())

print("B")
print(B)
U = B.reshape(N ** num)
Q = DEV2 @ U
C = Q.reshape([N] * num)
print("C")
print(C)
