# The Variational Principle in Quantum Mechanics:
---
The Variational Principle in Quantum Mechanics states that:

![equation](https://latex.codecogs.com/svg.latex?<\psi&space;|&space;\hat{H}&space;|&space;\psi>&space;=&space;E&space;=&space;\frac{\int_{-\infty}^{\infty}\psi^*&space;\hat{H}&space;\psi&space;d\bar{r}}{\int_{-\infty}^{\infty}\psi^*&space;\psi&space;d\bar{r}}&space;\ge&space;E_0)

---

This implementation utilises NumPy and Matplotlib to numerically calculate the energy eigenstates and energy eigenvalues of the given bounded potential system.

The code is capable of calculating the bound energies and states of any number of energy eigenstates, with decreasing accuracy with each new state.

The code can handle systems of any number of dimensions, but can only plot the energy eigenstates of a system up to and including 3D.

---

For full details read ![the report](report.pdf)
