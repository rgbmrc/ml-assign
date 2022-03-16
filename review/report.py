# %%
# region imports & defs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from simsio import *
import mplotter as plotter

from matplotlib.backend_bases import register_backend

register_backend("pdf", "matplotlib.backends.backend_pgf")
plotter.use_style(["mplotter/styles/rgbmrc.mplstyle", "review/report.mplstyle"])


@sim_or_uid_arg
def extract(sim, metric, epochs=1):
    """
    Extract simulation data.
    """
    dat = []
    samples = sim.par["samples"]
    for s in range(samples):
        key = f"history_{s}"
        sim.link(key, serializer="simsio.serializers.NPZSerializer")
        dat.append(sim.load(key)[metric])
    dat = np.asarray(dat)[:, -epochs:]
    # exp = np.median(dat)
    # dev = np.median(abs(dat - exp))
    exp = np.mean(dat)
    dev = np.std(dat)
    return exp, dev, sim.par["monitoring"]["cpu_time"] / samples


def collect(group, grid_keys):
    match = SimsQuery(group)
    ugrid, vals = uids_grid(match.uids, grid_keys)
    dat = np.vectorize(extract)(ugrid, metric="val_accuracy", epochs=1)
    return ugrid, vals, dat


def gridplot(dat, vals, slc, par):
    """
    Plot simulation grid.
    """
    fig = plt.figure(
        ",".join((p.rsplit("/")[-1] for p in par)), figsize=plotter.fig_size(2, 0.15)
    )
    imgrid = ImageGrid(
        fig,
        111,
        (1, len(dat)),
        axes_pad=0.50,
        cbar_pad=0.05,
        cbar_size=0.06,
        share_all=True,
        cbar_mode="each",
        label_mode="L",
    )
    for ax, l, d in zip(
        imgrid, ("accuracy (mean)", "accuracy (std. dev.)", "train CPU-time [s]"), dat
    ):
        sm = ax.imshow(d[slc], origin="lower", cmap="magma")
        for axis, p in zip((ax.yaxis, ax.xaxis), par):
            vs = vals[p]
            axis.set_major_locator(mpl.ticker.FixedLocator(np.arange(len(vs))))
            axis.set_major_formatter(mpl.ticker.FixedFormatter(vs))
            # axis.set_label_text(p)
        cbar = ax.cax.colorbar(sm)
        cbar.set_label(l)
    return fig, imgrid


# endregion

# %%
# region [1]

ugrid, vals, dat = collect(
    "1", ["input/N", "input/train_frac", "input/augment_frac", "input/augment_std"]
)

# %% plots
fig, imgrid = gridplot(dat, vals, np.s_[:, :, 0, 0], ["input/N", "input/train_frac"])
for ax in imgrid.axes_row[-1]:
    ax.set_xlabel("$\chi$")
for ax in imgrid.axes_column[0]:
    ax.set_ylabel("$N$")
# plotter.set_fig_size(fig)
plotter.save_fig(fig)

# %%
fig, imgrid = gridplot(
    dat, vals, np.s_[-3, :, :, -1], ["input/train_frac", "input/augment_frac"]
)
for ax in imgrid.axes_row[-1]:
    ax.set_xlabel("$\eta$")
for ax in imgrid.axes_column[0]:
    ax.set_ylabel("$\chi$")
# plotter.set_fig_size(fig)
plotter.save_fig(fig)

# endregion

# %%
# region [2A]

ugrid, vals, dat = collect("2A", ["compile/optimizer", "model/activation"])

exp, std, cpu = dat
i, j = np.unravel_index(exp.argmax(), ugrid.shape)
arg_i, arg_j = np.argsort(exp[:, j]), np.argsort(exp[i])
ugrid = ugrid[arg_i[..., np.newaxis], arg_j[np.newaxis]]
vals["compile/optimizer"] = vals["compile/optimizer"][arg_i]
vals["model/activation"] = vals["model/activation"][arg_j]

dat = np.vectorize(extract)(ugrid, metric="val_accuracy", epochs=1)

# %% plots
fig, imgrid = gridplot(
    dat[:2], vals, np.s_[:, :], ["compile/optimizer", "model/activation"]
)

for ax in imgrid.axes_row[-1]:
    ax.set_xlabel("activation")
    ax.set_xticklabels(vals["model/activation"], rotation=90)
for ax in imgrid.axes_column[0]:
    ax.set_ylabel("optimizer")
imgrid[0].get
# plotter.set_fig_size(fig)
plotter.save_fig(fig)

# endregion

# %%
# region [2B]

ugrid, vals, dat = collect("2B", ["input/N", "model/M"])

# %% plots
fig, imgrid = gridplot(dat[:2], vals, np.s_[:, :], vals.keys())

for ax in imgrid.axes_row[-1]:
    ax.set_xlabel("$M$")
for ax in imgrid.axes_column[0]:
    ax.set_ylabel("$N$")
# plotter.set_fig_size(fig)
plotter.save_fig(fig)

# endregion

# %%
# region [2C]

ugrid, vals, dat = collect("2C", ["model/dropout", "model/layers/1"])

# %% plots
fig, imgrid = gridplot(dat[:2], vals, np.s_[:, :], vals.keys())

for ax in imgrid.axes_row[-1]:
    ax.set_xticklabels(np.arange(1, 5))
    ax.set_xlabel("hidden layers")
for ax in imgrid.axes_column[0]:
    ax.set_ylabel("dropout")
# plotter.set_fig_size(fig)
plotter.save_fig(fig, "dropout,layers")

# endregion

# %%
