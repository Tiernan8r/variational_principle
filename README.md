The Variational Principle in Quantum Mechanics:
===
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Tiernan8r_variational_principle&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Tiernan8r_variational_principle)

The Variational Principle in Quantum Mechanics states that:

![equation](https://latex.codecogs.com/svg.latex?<\psi&space;|&space;\hat{H}&space;|&space;\psi>&space;=&space;E&space;=&space;\frac{\int_{-\infty}^{\infty}\psi^*&space;\hat{H}&space;\psi&space;d\bar{r}}{\int_{-\infty}^{\infty}\psi^*&space;\psi&space;d\bar{r}}&space;\ge&space;E_0)

---

This implementation utilises NumPy and Matplotlib to numerically calculate the energy eigenstates and energy eigenvalues of the given bounded potential system.

The code is capable of calculating the bound energies and states of any number of energy eigenstates, with decreasing accuracy with each new state.

The code can handle systems of any number of dimensions, but can only plot the energy eigenstates of a system up to and including 3D.

---

For full details read ![the report](report.pdf)

### Setup:

First, setup a virtualenvironment and install the requirements:
```console
$ python -m virtualenv venv
```
```console
$ source venv/bin/activate
```
```console
$ pip install -r requirements.txt
```

### Running:

Ensure the setup steps above are run.

From the top level directory:
```console
$ python -m variational_principle.main
```

Should result in a help message like the following:
```console
main.py - A tool to calculate the n energy eigenstates for a given bound potential
USAGE:
main.py [FLAGS] <start> <stop> <N>
FLAGS:
-i/--include-potential       Whether or not to plot the calculated graphs with the given potential superimposed
                             (default = False)
-v/--v-scale                 Scaling factor used on the potential when --include-potential is True (default = 10)
-d/--dimensions              The number of dimensions to calculate on, options are 1, 2 or 3 (default = 1)
-n/--num-states              Number of energy eigenstates to calculate (default = 1)
-n/--num-iterations          The number of iterations to calculate (default = 10^5)
-h/--help                    Show this message
ARGUMENTS:
<start>                      The initial coordinate for the grid
<stop>                       The end coordinate for the grid
<N>                          The number of subdivisions of the grid
```

To verify the code will run correctly, the easiest input is:
```console
$ python -m variational_principle.main -1 1 1
```

If no errors occur, a GUI should appear with an empty plot.

After that, happy plotting!

### Installing:

Using `pyinstaller` an executable binary can be compiled:

```console
$ pyinstaller --onefile binary.spec
```
Will compile an executable binary at `dist/variational_principle`.
Run the binary as usual with
```console
$ ./dist/variational_principle [...input]
```
