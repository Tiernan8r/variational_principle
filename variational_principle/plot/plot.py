import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import vstack

# the colour map for the surface plots.
colour_map = "autumn"
# The colour map is set by the global string
cmap = plt.cm.get_cmap(colour_map)

# neatness nicety, for displaying indexing of the states.
th = {1: "st", 2: "nd", 3: "rd"}

# a list of names for the 1st 10 axes.
axes = ("x", "y", "z", "w", "q", "r", "s", "t", "u", "v")

# The name of the QM system to be shown in the plot titles.
pot_sys_name = "Harmonic Oscillator"


def plotting(r, all_psi, D, include_V=False, V=None, V_scale=1):
    """
    A method that contains various forms of plotting of psi and the potential of the system.
    :param r: The grid coordinates.
    :param all_psi: All the wavefunctions to plot.
    :param D: The number of axes in the system.
    :param include_V: Whether to plot the potential or not.
    :param V: The potential to plot if so.
    """

    # If the system is 1D, plot a line
    if D == 1:

        # The number of psi states to plot.
        num_states = len(all_psi)

        # Generate the legend for the system, naming the 0th state as the Ground State.
        state_names = ["Ground State"]
        for i in range(1, num_states):
            state_names += ["{}{} State".format(i, th.get(i, "th"))]

        # If we want to plot the potential, plot it first.
        # all_psi = np.vstack(([V], all_psi))
        all_psi = vstack(([V], all_psi))
        state_names = ["Potential"] + state_names

        # iterate over all the functions to plot, and plot them individually.
        for i in range(len(all_psi)):
            title = "The {} for the {} along $x$:".format(state_names[i], pot_sys_name)
            file_name = "state_{}".format(i - 1)
            state = "$\psi$"

            plt_with_V = include_V
            with_V_scale = V_scale
            if state_names[i] == "Potential":
                file_name = "potential"
                state = "V"
                plt_with_V = False
                with_V_scale = 1

            legend = [state_names[i]]
            if plt_with_V:
                legend = [state_names[0]] + legend
            legend = tuple(legend)

            plot_line(*r, all_psi[i], title, state, legend=legend, include_V=plt_with_V, V=V, V_scale=with_V_scale,
                      filename=file_name)

    # If the system is 2D, plot the img, wireframe and surfaces.
    elif D == 2:

        if include_V:
            title = "The Potential function for the {} along $x$ & $y$".format(pot_sys_name)
            plot_img(*r, V, title)
            plot_wireframe(*r, V, title, "V")
            plot_surface(*r, V, title, "V")

        num_states = len(all_psi)
        for n in range(num_states):
            title = "$\psi_{}$ for the {} along $x$ & $y$".format(n, pot_sys_name)
            plot_img(*r, all_psi[n], title)
            plot_wireframe(*r, all_psi[n], title, "$\psi$")
            plot_surface(*r, all_psi[n], title, "$\psi$")

    # if the system is 3D, plot the 3D scatter.
    elif D == 3:

        if include_V:
            title = "The Potential function for the {} along $x$, $y$ & $z$".format(pot_sys_name)
            N = V.shape[1]
            D = len(V.shape)
            V = V.reshape(N ** D)
            plot_3D_scatter(*r, V, title)

        num_states = len(all_psi)
        for n in range(num_states):
            title = "$\psi_{}$ for the {} along $x$, $y$ & $z$".format(n, pot_sys_name)

            N = all_psi[n].shape[1]
            D = len(all_psi[n].shape)
            psi = all_psi[n].reshape(N ** D)

            plot_3D_scatter(*r, psi, title)

    # All higher order systems can't be easily visualised.
    else:
        return


# A method to plot the 1D system as a line.
def plot_line(x, y, title, ylabel="$\psi$", legend=None, filename=None, include_V=False, V=None, V_scale=1):
    if include_V:
        plt.plot(x, V)
    plt.plot(x, y * V_scale)
    plt.xlabel("$x$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    if filename is not None:
        plt.savefig("../data/plots/" + filename)
    plt.show()


# A method to plot the 2D system as a flat image.
def plot_img(x, y, z, title):
    plt.contourf(x, y, z, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()


# A method to plot the 2D system as a wireframe.
def plot_wireframe(x, y, z, title, zlabel="$\psi$"):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_wireframe(x, y, z)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()


# A method to plot the 2D system as a surface plot.
def plot_surface(x, y, z, title, zlabel="$\psi$"):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(x, y, z, cmap=cmap)
    fig.colorbar(surf, ax=ax)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()


# A method to plot the 3d system as a 3D scatter plot.
def plot_3D_scatter(x, y, z, vals, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    p = ax.scatter3D(x, y, zs=z, c=vals, cmap=colour_map)
    fig.colorbar(p, ax=ax)

    plt.title(title)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.show()
