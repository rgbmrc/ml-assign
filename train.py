# %%
import pickle
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout

L = 2
EPOCHS = 200
N = 2 ** np.arange(0, 4) * 1000
perc_train = np.arange(0.5, 1, 0.1)
N = [4000]
perc_train = [0.8]

# %% data generation
def func(x):
    return (x[..., 0] > -20) & (x[..., 1] > -40) & ((x[..., 0] + x[..., 1]) < 40)


def generate(N, B=1, seed=False):
    # random seed for reproducibility
    if seed:
        np.random.seed(seed)
    x = (np.random.random((N, L)) - 0.5) * B
    y = func(x)
    return x, y


# %% models
models = [
    Sequential(
        layers=[
            # input layer
            Dense(L, activation="relu"),
            # hidden layers
            Dense(20, activation="relu"),
            Dense(20, activation="relu"),
            # dropout: fraction of nodes not working in each iteration,
            # to make the model less relying on individual nodes;
            # next line imposes dropout only on last layer TODO: ?
            Dropout(0.2),
            # output node for classification
            Dense(1, activation="sigmoid"),
        ],
        name="model",
    ),
]

# %% training
fits = {}
for mod, n, p in product(models, N, perc_train):
    fits[(mod.name, n, p)] = mod = clone_model(mod)
    mod.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    n_train = int(n * p)
    x, y = generate(n, B=100)
    x /= 50
    mod.fit(
        x[:n_train],
        y[:n_train],
        epochs=EPOCHS,
        batch_size=50,
        validation_data=(x[n_train:], y[n_train:]),
        verbose=0,
    )


# %% loss vs epoch
for f in fits.values():
    fig, ax = plt.subplots()
    ax.plot(f.history.history["loss"], label="training")
    ax.plot(f.history.history["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()

# %%
obs = "val_accuracy"
plt.plot(perc_train, [f.history.history[obs][-1] for f in fits.values()], "o-")
plt.ylabel(obs)
plt.xlabel("perc_train")
plt.grid()

# %%
lims = xmin, xmax = x.min(), x.max()
grid = np.mgrid[xmin:xmax:100j, xmin:xmax:100j]
_, mod = fits.copy().popitem()
pred = mod.predict(grid.T)

fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
imkws = dict(origin="lower", extent=lims * 2)
axs[0].scatter(*x.T, c=y)
axs[0].margins(x=0, y=0)
axs[1].imshow(pred, **imkws)
axs[2].imshow(pred > 0.5, **imkws)

# %%
