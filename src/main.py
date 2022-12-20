#!/usr/bin/env python

import sys
import os

# Required to guarantee that the 'src' module is accessible when
# this file is run directly.
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import re
from typing import Tuple

import variational_principle.compute as comp
import variational_principle.plot as plt


def usage():
    print(sys.argv[0], "- A tool to calculate the n energy eigenstates for a given bound potential")
    print("USAGE:")
    print(sys.argv[0], "[FLAGS] <start> <stop> <N>")
    print("FLAGS:")
    print(
        "-i/--include-potential       Whether or not to plot the calculated graphs with the given potential superimposed")
    print("                             (default = False)")
    print(
        "-v/--v-scale                 Scaling factor used on the potential when --include-potential is True (default = 10)")
    print("-d/--dimensions              The number of dimensions to calculate on, options are 1, 2 or 3 (default = 1)")
    print("-n/--num-states              Number of energy eigenstates to calculate (default = 1)")
    print("-n/--num-iterations          The number of iterations to calculate (default = 10^5)")
    print("-h/--help                    Show this message")
    print("ARGUMENTS:")
    print("<start>                      The initial coordinate for the grid")
    print("<stop>                       The end coordinate for the grid")
    print("<N>                          The number of subdivisions of the grid")
    sys.exit(0)


# Parses the cli input and returns the following values:
# start, stop, N, num_dimensions, num_states, num_iterations, include_potential, v_scale.
def parse_input() -> Tuple[int, int, int, int, int, int, bool, int]:
    if len(sys.argv) < 4:
        usage()

    # Will be overwritten
    start, stop, N = -10, 10, 100
    # The stated defaults
    num_dimensions = 1
    num_states = 1
    num_iterations = 10 ** 5
    include_potential = False
    v_scale = 10

    values = []
    arguments = sys.argv[1:]
    i = 0
    for a in arguments:
        arg = arguments[i]

        if arg == "-h" or arg == "--help":
            usage()

        elif arg == "-i" or arg == "--include-potential":
            include_potential = True

        elif arg == "-v" or arg == "--v-scale":
            if len(arguments) <= i + 1:
                print("Missing value for flag '--v-scale'")
                sys.exit(1)
            v_scale = int(arguments[i + 1])
            # Remove the value just read from the args so we don't iterate over it
            del arguments[i + 1]

            if v_scale < 0:
                print(f"Invalid value '{v_scale}' for '--v-scale'")

        elif arg == "-d" or arg == "--dimensions":
            if len(arguments) <= i + 1:
                print("Missing value for flag '--dimensions'")
                sys.exit(1)
            num_dimensions = int(arguments[i + 1])
            del arguments[i + 1]

            if num_dimensions < 1 or num_dimensions > 3:
                print(f"Invalid value '{num_dimensions}' for '--num-dimensions")
                usage()

        elif arg == "-s" or arg == "--num-states":
            if len(arguments) <= i + 1:
                print("Missing value for flag '--num-states'")
                sys.exit(1)
            num_states = int(arguments[i + 1])
            del arguments[i + 1]

            if num_states < 1:
                print(f"Num states must be greater than zero.")
                usage()

        elif arg == "-n" or arg == "--num-iterations":
            if len(arguments) <= i + 1:
                print("Missing value for flag '--num-iterations'")
                sys.exit(1)
            num_iterations = int(arguments[i + 1])
            del arguments[i + 1]

            if num_iterations < 0:
                print(f"Invalid value '{num_iterations}' for '--num-iterations'")

        elif re.match("--.+", arg) or re.match("-.+", arg):
            # If the unrecognised value can be parsed as an int, it could be a negative number...
            try:
                tmp = int(arg)
                values.append(arg)
            except Exception as e:
                print("Unrecognised flag '{0}'".format(arg))
        else:
            values.append(arg)

        i += 1

    # Final sanity check on num values:
    if len(values) != 3:
        print("Arguments mismatch!")
        usage()

    start = int(values[0])
    stop = int(values[1])
    if stop < start:
        print(f"<start> must be less than <stop>, given '{start}' '{stop}'")
        usage()

    N = int(values[2])
    if N < 1:
        print(f"<N> must be greater than zero, given '{N}'")
        usage()

    return start, stop, N, num_dimensions, num_states, num_iterations, include_potential, v_scale


def main():
    start, stop, N, num_dimensions, num_states, num_iterations, include_potential, v_scale = parse_input()
    print(f"<start> is: {start}")
    print(f"<stop> is: {stop}")
    print(f"<N> is: {N}")
    print("Number of dimensions:", num_dimensions)
    print("Number of iterations:", num_iterations)
    print("Include potential?", include_potential)
    print("Potential scaling factor:", v_scale)
    print("------------------------")

    print(f"Beginning calculation over '{num_iterations}' iterations in {num_dimensions}D.")

    r, V, all_psi = comp.compute(start, stop, N, num_dimensions, num_states, num_iterations)

    print("Finished computations, generating plots...")
    if include_potential:
        print(f"Plots will include potential, scaled by '{v_scale}'")

    # plot the generated psis.
    plt.plot.plotting(r, all_psi, num_dimensions, include_potential, V, v_scale)


if __name__ == "__main__":
    main()
