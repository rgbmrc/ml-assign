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
    # exp = np.median(dat)
    # dev = np.median(abs(dat - exp))
    exp = np.mean(dat)
    dev = np.std(dat)
    return exp, dev, sim.par["monitoring"]["cpu_time"]


dat = np.vectorize(extract)(ugrid, "val_accuracy", epochs=1)

# %% plot
def gridplot(slc, par):
    fig = plt.figure(figsize=(15, 5))
    imgrid = ImageGrid(
        fig,
        111,
        (1, 3),
        axes_pad=0.85,
        cbar_pad=0.15,
        share_all=True,
        cbar_mode="each",
    )
    for ax, l, d in zip(imgrid, ("median", "std. dev.", "cpu time"), dat):
        sm = ax.imshow(d[slc], origin="lower", cmap="magma")
        for axis, p in zip((ax.yaxis, ax.xaxis), par):
            v = vals[p]
            axis.set_major_locator(mpl.ticker.FixedLocator(np.arange(len(v))))
            axis.set_major_formatter(mpl.ticker.FixedFormatter(v))
            axis.set_label_text(p)
        cbar = ax.cax.colorbar(sm)
        cbar.set_label(l)
    return fig, imgrid


# %% plots
fig, _ = gridplot(np.s_[:, :, 0, 0], ["input/N", "input/train_frac"])
fig.savefig("N*train_frac.png")
fig, _ = gridplot(np.s_[-3, :, :, -1], ["input/train_frac", "input/augment_frac"])
fig.savefig("train_frac*augment_frac.png")

# %% ditributions
metric = "val_accuracy"
epochs = 5
hist_dat = []
for u in ugrid[-2, 1, 0, :]:
    sim = get_sim(u)
    for s in range(sim.par["samples"]):
        key = f"history_{s}"
        sim.link(key, serializer="simsio.serializers.NPZSerializer")
        hist_dat.append(sim.load(key)[metric])
hist_dat = np.asarray(hist_dat)[:, -epochs:]
plt.hist(hist_dat.ravel())

# %%
