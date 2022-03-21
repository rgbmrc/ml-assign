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

labels = {
    "input/N": "$N$",
    "input/train_frac": "$\chi$",
    "input/augment_frac": "$\eta$",
    "model/vars/activation": "activation",
    "model/vars/M": "$M$",
    "model/vars/dropout": "dropout",
    "model/layers": "hidden layers",
    "compile/optimizer": "optimizer",
    "input/rescale": "rescaling",
    "compile/optimizer/config/learning_rate": "learning rate",
}


@sim_or_uid_arg
def extract_all(sim, metric="val_accuracy"):
    """
    Extract simulation data.
    """
    dat = []
    samples = sim.par["samples"]
    for s in range(samples):
        key = f"history_{s}"
        sim.link(key, serializer="simsio.serializers.NPZSerializer")
        dat.append(sim.load(key)[metric])
    return np.asarray(dat)


@sim_or_uid_arg
def extract_averages(sim, metric, epochs=1):
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


def compute(sim):
    # TODO:
    raise NotImplementedError
    dat = []
    x = np.random.random((N, 2)) * 100 - 50
    y = (x[..., 0] > -20) & (x[..., 1] > -40) & ((x[..., 0] + x[..., 1]) < 40)
    samples = sim.par["samples"]
    sim.link("model", serializer="simsio.serializers.KerasModelSerializer")
    mod = sim.load("model")
    for s in range(samples):
        key = f"weights_{s}"
        sim.link(key, serializer="simsio.serializers.KerasWeightSerializer")
        mod.load_weights(sim.handles[key].storage)
        model.evaluate(x, y)


def collect(group, grid_keys):
    match = SimsQuery(group)
    ugrid, vals = uids_grid(match.uids, grid_keys)
    dat = np.vectorize(extract_averages)(ugrid, metric="val_accuracy")
    return ugrid, vals, dat


def gridplot(dat, vals):
    """
    Plot simulation grid.
    """
    figname = ",".join((p.rsplit("/")[-1] for p in vals))
    padsize = 0.09
    figsize = padsize * (len(dat) - 1) + 0.25 * len(dat) * dat[0].shape[-1]
    fig = plt.figure(figname, figsize=(figsize, 9))
    imgrid = ImageGrid(
        fig,
        111,
        (1, len(dat)),
        axes_pad=padsize,
        cbar_pad=padsize,
        cbar_size=0.06,
        share_all=True,
        cbar_mode="each",
        cbar_location="top",
        label_mode="L",
    )
    for ax, d, l, cmap in zip(
        imgrid,
        dat,
        ("accuracy ($\mu$)", "accuracy ($\sigma$)", "train time [s]"),
        ("magma", "magma_r", "magma_r"),
    ):
        avg, dev = d.mean(), 2 * d.std()
        norm = mpl.colors.Normalize(max(avg - dev, d.min()), min(avg + dev, d.max()))
        sm = ax.imshow(d, origin="lower", cmap=cmap, norm=norm)
        for axis, p in zip((ax.yaxis, ax.xaxis), vals):
            vs = vals[p]
            axis.set_major_locator(mpl.ticker.FixedLocator(np.arange(len(vs))))
            axis.set_major_formatter(mpl.ticker.FixedFormatter(vs))
            axis.set_label_text(labels.get(p, p))
        cbar = ax.cax.colorbar(sm)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.set_label(l)
    return fig, imgrid


# endregion

# %%
# region [1A]
ugrid, vals, dat = collect("II/1A", ["input/N", "input/train_frac"])

# %% plots
fig, imgrid = gridplot(dat, vals)
plotter.save_fig(fig)

# endregion

# %%
# region [1B]
ugrid, vals, dat = collect("II/1B", ["input/augment_frac", "input/train_frac"])
_ugrid, _vals = uids_grid(SimsQuery("II/1A").uids, ["input/N", "input/train_frac"])
_dat = np.vectorize(extract_averages)(_ugrid[-2], "val_accuracy")
dat = [np.vstack([d, _d]) for (d, _d) in zip(dat, _dat)]
vals["input/augment_frac"] = [f"${l}$" for l in ("0", "1/3", "2/3", "1", "(\star)")]

# %%
fig, imgrid = gridplot(dat, vals)
imgrid.cbar_axes[1].set_xticks([0.025, 0.035])
plotter.save_fig(fig)

# endregion

# %%
# region [2A]
ugrid, vals, dat = collect("II/2A", ["compile/optimizer", "model/vars/activation"])

exp, std, cpu = dat
i, j = np.unravel_index(exp.argmax(), ugrid.shape)
arg_i, arg_j = np.argsort(exp[:, j]), np.argsort(exp[i])
ugrid = ugrid[arg_i[..., np.newaxis], arg_j[np.newaxis]]
vals["compile/optimizer"] = vals["compile/optimizer"][arg_i]
vals["model/vars/activation"] = vals["model/vars/activation"][arg_j]

dat = np.vectorize(extract_averages)(ugrid, metric="val_accuracy", epochs=1)

# %% plots
fig, imgrid = gridplot(dat[:2], vals)
for ax in imgrid.axes_row[-1]:
    ax.xaxis.set_tick_params(rotation=90)
# plotter.set_fig_size(fig, plotter.fig_size(0.85, 0.20))
plotter.save_fig(fig)

# endregion

# %%
# region [2B]
ugrid, vals, dat = collect("II/2B", ["input/N", "model/vars/M"])

# %% plots
fig, imgrid = gridplot(dat[:2], vals)
# plotter.set_fig_size(fig, plotter.fig_size(0.75, 0.15))
plotter.save_fig(fig)

# endregion

# %%
# region [2C]
layers = lambda p: (len(p["model"]["layers"]) - 2) // 2
ugrid, vals, dat = collect("II/2C", ["model/vars/dropout", layers])
vals["model/layers"] = vals.pop(layers)

# %% plots
fig, imgrid = gridplot(dat[:2], vals)
plotter.save_fig(fig)

# endregion

# %%
# region [3A]
ugrid, vals, dat = collect(
    "II/3A", ["input/rescale", "compile/optimizer/config/learning_rate"]
)

# %% plots
fig, imgrid = gridplot(dat[:2], vals)
for ax in imgrid:
    ax.set_xticklabels(["", "$10^{-3}$", "", "$10^{-2}$", ""])
    ax.set_yticklabels(["$10^{0}$", "$10^{1}$", "$10^{2}$", "$10^{3}$"])
plotter.save_fig(fig)


# endregion

# %%
# region [GLOBAL]

match = SimsQuery("II/*")
uids = list(match.uids)


# %%
# region histogram
accu = np.concatenate([extract_all(u)[..., -1] for u in uids])

# %%
fig, ax = plt.subplots(figsize=plotter.fig_size(0.75, ratio=0.667))
ax.hist(accu, bins=np.linspace(0.88, 1, 49), fc="k")
ax.margins(y=0.1)
ax.set_xlabel("accuracy")
ax.set_ylabel("realizations")
plotter.set_fig_size(fig)
plotter.save_fig(fig, "accuracy")

# endregion

# %%
# region learning curve

data = extract_all("9fb5fc38a72111eca9d0fa163e209104", "val_accuracy")

fig, ax = plt.subplots(1, 1, figsize=plotter.fig_size(0.6, ratio=0.6))
x = np.arange(data.shape[-1]) + 0.5
y = 1 - data.mean(0)
dy = data.std(0)
ax.fill_between(x, y - dy, y + dy, fc="0.8")
ax.plot(x, y, c="k")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("$1-{}$accuracy$")
plotter.save_fig(fig, "curve")

# endregion

# %%
# region best triangle

_loss = [extract_all(u, "val_loss")[..., -1] for u in uids]

loss_argmin = [np.argmin(l) for l in _loss]
max_samples = max(len(l) for l in _loss)
loss = np.asarray(
    [
        np.pad(
            l,
            (0, max_samples - len(l)),
            "constant",
            constant_values=np.nan,
        )
        for l in _loss
    ]
)
u, s = np.unravel_index(np.nanargmin(loss), loss.shape)
u = uids[u]
u, s


# %%
sim = get_sim(u)
key = f"weights_{s}"
sim.link("model", serializer="simsio.extensions.ext_keras.KerasModelSerializer")
sim.link(key, serializer="simsio.extensions.ext_keras.KerasWeightSerializer")
mod = sim.load("model")
mod.load_weights(sim.handles[key].storage)
ext = 1
grid = np.mgrid[-ext:ext:100j, -ext:ext:100j]

x = grid.T.reshape(-1, 2)
pred = mod.predict(x).reshape(100, -1)

fig = plt.figure("triangle", figsize=plotter.fig_size(0.75, ratio=1))
imgrid = ImageGrid(
    fig,
    111,
    (1, 1),
    axes_pad=0.09,
    cbar_pad=0.09,
    cbar_size=0.06,
    share_all=True,
    cbar_mode="each",
    cbar_location="right",
    label_mode="L",
)
ax = imgrid[0]
ext = np.array([-1, 1, -1, 1]) * 50

imkws = dict(origin="lower", extent=ext, cmap="coolwarm")
sm = ax.imshow(pred, **imkws)
lnkws = dict(ls=(0, (4, 6)), c="k")
ax.plot((-20, -20), (-40, 50), **lnkws)
ax.plot((-20, 50), (-40, -40), **lnkws)
ax.plot((-10, 50), (50, -10), **lnkws)
cbar = ax.cax.colorbar(sm)
cbar.set_label("$\hat{y}$", rotation=0, va="center", ha="left")

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$", rotation=0, va="center", ha="right")

locator = mpl.ticker.MultipleLocator(20)
formatter = plotter.annotating.SSDecimalFormatter(0, sign_zero=False)
ax.xaxis.set_major_locator(locator)
ax.yaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plotter.set_fig_size(fig)
plotter.save_fig(fig)

# endregion

# %%
# region count NN params

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def params(*Ns):
    mod = Sequential([Dense(2, input_shape=(2,))] + [Dense(n) for n in Ns] + [Dense(1)])
    return mod.count_params()


# endregion

# endregion

# %%
