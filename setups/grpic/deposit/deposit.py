import nt2.read as nt2r
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data = nt2r.Data("deposit.h5")
def frame(ti):
    t0 = data.t.isel(t=ti).values[()]
    vmax = 1e-5
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    data.Dr.isel(t=ti).polar.pcolor(
        ax=ax,
        norm=mpl.colors.SymLogNorm(
            vmin=-vmax, vmax=vmax, linthresh=vmax / 1e3, linscale=1
        ),
        cmap="RdBu_r",
    )
    x1, x2 = data.particles[1].isel(t=ti).r * np.sin(
        data.particles[1].isel(t=ti).th
    ), data.particles[1].isel(t=ti).r * np.cos(data.particles[1].isel(t=ti).th)
    ax.scatter(x1, x2, s=5, c="b", ec="w", lw=0.5)
    x1, x2 = data.particles[2].isel(t=ti).r * np.sin(
        data.particles[2].isel(t=ti).th
    ), data.particles[2].isel(t=ti).r * np.cos(data.particles[2].isel(t=ti).th)
    ax.scatter(x1, x2, s=5, c="r", ec="w", lw=0.5)
    ax.add_artist(
        mpl.patches.Arc(
            (0, 0),
            2 * 10,
            2 * 10,
            fill=False,
            ec="k",
            ls=":",
            lw=0.5,
            theta1=-90,
            theta2=90,
        )
    )
    axins = ax.inset_axes([0.4, 0.8, 0.6 - 0.025, 0.2 - 0.01])
    (
        data.Dr.sel(t=slice(data.t.min(), t0)).sel(r=10, method="nearest")
        * np.sin(data.th)
    ).sum(dim="th").plot(ax=axins)
    axins.set(
        ylim=(-3 * vmax, 3 * vmax),
        xlim=(data.t.min(), data.t.max()),
        title="",
        ylabel=rf"$\int D_r d\Omega$ @ (r = 10)",
    )
    axins.axvline(t0, color="b", lw=0.5, ls=":")
    axins.axhline(0, color="r", lw=0.5, ls="--")

frame(30)