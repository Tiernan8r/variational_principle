import numpy as np

def potential(r: np.ndarray) -> np.ndarray:
    """
    The potential energy function of the system
    :param r: The coordinate grid of the system for each axis.
    :return: The potential function V as a grid of values for each position.
    """

    N = r.shape[1]
    D = r.shape[0]

    # # Infinite Sq Well
    # #V_0 = np.inf
    # # Finite Sq Well
    # V_0 = 10
    #
    # third = N // 3
    #
    # addition = int(abs((third - (N / 3))) * 3)
    #
    # mid, bef = np.zeros(third + addition), np.linspace(V_0, V_0, third)
    # aft = bef.copy()
    # x_well = np.concatenate((bef, mid, aft))
    # wells = [x_well] * D
    # V = np.array(np.meshgrid(*wells, indexing="ij"))
    #
    # # Correction of Corners:
    # V = np.sum(V, axis=0)
    # V = V.reshape(N ** D)
    # # Perturbation
    # R = np.sum(r, axis=0)
    # R = R.reshape(N ** D)
    # for i in range(len(V)):
    #     if V[i] > 0:
    #         V[i] = V_0
    #     # Perturbation
    #     elif V[i] == 0:
    #         V[i] += 0.5 * R[i]
    # V = V.reshape([N] * D)

    # # Alpha Barrier Penetration
    # V = r.copy()
    # for i in range(len(r)):
    #     length = len(r[i])
    #     third = length // 3
    #
    #     Z = 48
    #     e = 1.6021753*10**-19
    #     e_0 = 8.854187817*10**-12
    #     pi = np.pi
    #
    #     # # Rc = 1.2 A^1.3 fm
    #     # Rc = r[i][third]
    #     # A = (Rc / 1.2) ** (1/1.3) * 10**-15
    #
    #     # U = -np.array([2.4 * (Z - 2) * A **-(1/3)] * third)
    #     U = -np.array([0.05] * third)
    #     U[0] = np.inf
    #
    #     # inv_r = ((Z - 2) * 2 * e**2) / (4 * pi * e_0 * r[i][third:])
    #     inv_r = 1 / r[i][third:]
    #
    #     V[i] = np.concatenate((U, inv_r))
    # V = np.sum(V, axis=0)

    # Harmonic Oscillator:
    V = 0.5 * r ** 2
    V = np.sum(V, axis=0)

    # # Anharmonic Oscillator:
    # V = r + 0.5 * r ** 2 + 0.25 * r **4
    # V = np.sum(V, axis=0)

    # # V-shaped
    # V = abs(r)
    # V = np.sum(V, axis=0)

    # # Inverse R:
    # V = r ** -1
    # V = np.sum(V, axis=0)

    # # Inverse Sq Law
    # V = r ** -2
    # V = np.sum(V, axis=0)

    # # Free Particle
    # V = np.zeros(r.shape)
    # V = np.sum(V, axis=0)

    # # # Central Potential:
    # A = -10
    # B = 1.5
    # C = 8
    #
    # V_c = A * C * r**-1
    # V_f = B * C**2 * r**-2
    #
    # V = V_c + V_f
    # V = np.sum(V, axis=0)

    # # Crystal Band Structure:
    # num_bands = 5
    # band_spacing = N // (2 * num_bands)
    # V_0 = 10
    #
    # x_band = np.linspace(V_0, V_0, N)
    # # for i in range(1, 2 * num_bands, 2):
    # i = 0
    # j = -1
    # for k in range(N):
    #     i %= band_spacing
    #     j %= 2
    #     if j == 0:
    #         x_band[k] = 0
    #     if i == 0:
    #         j += 1
    #     i += 1
    #
    # # for i in range(num_bands + 2):
    # #     start_index = band_spacing * i * 2
    # #     for j in range(band_spacing):
    # #         x_band[start_index + j] = 0
    #
    # wells = [x_band] * D
    # V = np.array(np.meshgrid(*wells, indexing="ij"))
    #
    # # Correction of Corners:
    # V = np.sum(V, axis=0)
    # V = V.reshape(N ** D)
    # # # Perturbation
    # # R = np.sum(r, axis=0)
    # # R = R.reshape(N ** D)
    # for i in range(len(V)):
    #     if V[i] > 0:
    #         V[i] = V_0
    #     # # Perturbation
    #     # elif V[i] == 0:
    #     #     V[i] += 0.5 * R[i]
    # V = V.reshape([N] * D)

    # # Delta Barrier:
    # V = np.zeros(r.shape)
    # sub_V = V
    # while True:
    #     sub_sub_V = sub_V[0]
    #     if not type(sub_sub_V) is np.ndarray:
    #         L = len(sub_V)
    #         sub_V[L // 2] = -np.inf
    #         break
    #     else:
    #         sub_V = sub_sub_V
    # V = V.sum(axis=0)

    return V
