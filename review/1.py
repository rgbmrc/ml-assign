# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from simsio import *


# %% collect runs
match = SimsQuery("1")
ugrid, vals = uids_grid(
    match.uids,
    ["input/N", "input/train_frac", "input/augment_frac", "input/augment_std"],
)

# %% extract data
@sim_or_uid_arg
def extract(sim, metric, epochs=1):
    dat = []
    for s in range(sim.par["samples"]):
        key = f"history_{s}"
        sim.link(key, serializer="simsio.serializers.NPZSerializer")
        dat.append(sim.load(key)[metric])
    dat = np.asarray(dat)[:, -epochs:]
    return dat.mean(), dat.std(), sim.par["monitoring"]["cpu_time"]


dat = np.vectorize(extract)(ugrid, "val_accuracy", epochs=1)

# %% plot
def gridplot(slc, par):
    fig = plt.figure(figsize=(20, 5))
    imgrid = ImageGrid(
        fig,
        111,
        (1, 3),
        axes_pad=0.85,
        cbar_pad=0.15,
        share_all=True,
        cbar_mode="each",
    )
    for ax, l, d in zip(imgrid, ("mean", "std. dev.", "cpu time"), dat):
        sm = ax.imshow(d[slc], origin="lower", cmap="magma")
        for coord, p in zip(("y", "x"), par):
            v = vals[p]
            getattr(ax, f"set_{coord}ticks")(np.arange(len(v)))
            getattr(ax, f"set_{coord}ticklabels")(np.round(v, 1))
            getattr(ax, f"set_{coord}label")(p)
        cbar = ax.cax.colorbar(sm)
        cbar.set_label(l)
    return fig, imgrid


# %% plots
fig, _ = gridplot(np.s_[:, :, 0, 0], ["input/N", "input/train_frac"])
fig.savefig("N*train_frac.png")
fig, _ = gridplot(np.s_[-2, :, :, -1], ["input/train_frac", "input/augment_frac"])
fig.savefig("train_frac*augment_frac.png")
