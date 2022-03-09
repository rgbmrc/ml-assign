# %%
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout

L = 2
EPOCHS = 100
N = 2 ** np.arange(0, 2) * 1000
perc_train = (0.5, 0.75)


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
            Dense(L, input_shape=(L,), activation="relu"),
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
    x, y = generate(n)
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
