import nt2.read as nt2r
import matplotlib.pyplot as plt
import matplotlib as mpl

data = nt2r.Data("shock.h5")


def frame(ti, f):
    quantities = [
        {
            "name": "density",
            "compute": lambda f: f.N_2 + f.N_1,
            "cmap": "inferno",
            "norm": mpl.colors.Normalize(0, 5),
        },
        {
            "name": r"$E_x$",
            "compute": lambda f: f.Ex,
            "cmap": "RdBu_r",
            "norm": mpl.colors.Normalize(-0.05, 0.05),
        },
        {
            "name": r"$E_y$",
            "compute": lambda f: f.Ey,
            "cmap": "RdBu_r",
            "norm": mpl.colors.Normalize(-0.05, 0.05),
        },
        {
            "name": r"$E_z$",
            "compute": lambda f: f.Ez,
            "cmap": "RdBu_r",
            "norm": mpl.colors.Normalize(-0.05, 0.05),
        },
        {
            "name": r"$B_x$",
            "compute": lambda f: f.Bx,
            "cmap": "BrBG",
            "norm": mpl.colors.Normalize(-0.05, 0.05),
        },
        {
            "name": r"$B_y$",
            "compute": lambda f: f.By,
            "cmap": "BrBG",
            "norm": mpl.colors.Normalize(-0.05, 0.05),
        },
        {
            "name": r"$B_z$",
            "compute": lambda f: f.Bz,
            "cmap": "BrBG",
            "norm": mpl.colors.Normalize(-0.05, 0.05),
        },
    ]
    fig = plt.figure(figsize=(12, 5.5), dpi=300)
    gs = fig.add_gridspec(len(quantities), 1, hspace=0.02)
    axs = [fig.add_subplot(gs[i]) for i in range(len(quantities))]

    for ax, q in zip(axs, quantities):
        q["compute"](f.isel(t=ti)).plot(
            ax=ax,
            cmap=q["cmap"],
            norm=q["norm"],
            cbar_kwargs={"label": q["name"], "shrink": 0.8, "aspect": 10, "pad": 0.005},
        )
    for i, ax in enumerate(axs):
        ax.set(aspect=1)
        if i != 0:
            ax.set(title=None)
        if i != len(axs) - 1:
            ax.set(
                xticks=[],
                xticklabels=[],
                xlabel=None,
                title=ax.get_title().split(",")[0],
            )
    return fig
