import math
import numpy
import matplotlib.pyplot as plt
from scipy import integrate
from quantum_functions import *

# psi is complex
# psi = numpy.zeros(number_samples, dtype=numpy.complex)
# psi = numpy.random.normal((x_max - x_min) / 2, x_max / 4, number_samples)
psi = numpy.linspace(10, 10, number_samples)

E = energy_expectation(psi)
print(E)

plt.plot(x, potential(x))
plt.title("V")
plt.xlabel("x")
plt.ylabel("V")
plt.plot(x, psi)
plt.ylabel("$\psi$")
plt.show()


def tweak_psi(psi: numpy.ndarray, index: int, amount: float):
    new_psi = psi.copy()
    new_psi[index] += amount
    return new_psi


psi = numpy.linspace(10, 10, number_samples)

prev_E = 1 * 10 ** 1  # the previous energy value

total_loops = 0
E = energy_expectation(psi)
diff_E = prev_E - E

log = True
tweak = 8

keep_looping = True
while keep_looping:
    total_loops += 1
    diff_E = numpy.abs(prev_E - E)

    num_no_change = 0
    unchanged = True
    # Change psi

    for i in range(number_samples):

        # Tweak up & tweak down
        # psi_up = tweak_psi(psi, i, +1)
        # psi_down = tweak_psi(psi, i, -1)
        psi_up = tweak_psi(psi, i, +tweak)
        psi_down = tweak_psi(psi, i, -tweak)

        E_up = energy_expectation(psi_up)
        E_down = energy_expectation(psi_down)

        diff_up = numpy.abs(prev_E - E_up)
        diff_down = numpy.abs(prev_E - E_down)

        if log:
            count = str(total_loops) + "." + str(i) + ": "
            print(count, "psi_up: ", psi_up)
            print(count, "psi: ", psi)
            print(count, "psi_d: ", psi_down)

            print(count, "E_up: ", E_up)
            print(count, "E: ", E)
            print(count, "E_do: ", E_down)

            print(count, "dUp: ", diff_up)
            print(count, "dNow: ", diff_E)
            print(count, "dDwn: ", diff_down)

        # if the up makes a lower E diff, keep it
        if diff_up < diff_down and diff_up < diff_E:
            if log:
                print("\tCHOOSE UP")
            E = E_up
            diff_E = diff_up
            psi = psi_up
        # if down makes a lower E, keep it
        elif diff_down < diff_up and diff_down < diff_E:
            if log:
                print("\tCHOOSE DOWN")
            E = E_down
            diff_E = diff_down
            psi = psi_down
        # otherwise keep the current E
        else:
            if log:
                print("\tNO CHANGE: unchanged? ", unchanged)
            num_no_change += 1

        prev_E = E
        prev_psi = psi
        # E = energy_expectation(psi)

    keep_looping = diff_E > accuracy

    # If a full loop has been performed without any tweaking, reduce the tweak size
    # and re loop
    unchanged = num_no_change == number_samples
    # if unchanged:
    if num_no_change == number_samples:
        tweak /= 2.0
        keep_looping = True

    if log:
        print("#", total_loops, " Tweak: ", tweak)
        print("E diff in loop: ", diff_E)
        print("From sum: ", numpy.abs(prev_E - E))

if log:
    print("E diff:", diff_E)
    print("E:", E)
plt.plot(x, psi)
plt.title("$\psi$")
plt.xlabel("x")
plt.ylabel("$\psi$")
plt.show()
