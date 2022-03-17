# %%
# region defs

from collections import defaultdict

import numpy as np
import tensorflow as tf


def generate(N, box=1, train_frac=1.0, augment_frac=0.0, augment_std=0.1):
    """
    Data generation.
    """
    x = (np.random.random((N, 2)) - 0.5) * box
    y = (x[..., 0] > -20) & (x[..., 1] > -40) & ((x[..., 0] + x[..., 1]) < 40)
    x = (x - x.mean()) / x.std()
    N_train = int(N * train_frac)
    N_augment = int(N_train * augment_frac)
    x_train = np.pad(x[:N_train], ((0, N_augment), (0, 0)), mode="wrap")
    y_train = np.pad(y[:N_train], ((0, N_augment)), mode="wrap")
    valid = [a[N_train:] for a in (x, y)]
    x_train[:N_augment] += np.random.normal(0.0, augment_std, size=(N_augment, 2))
    return (x_train, y_train), valid


def train(params):
    samples = params.setdefault("samples", 1)
    results = defaultdict(list)

    for _ in range(samples):
        # input
        train, valid = generate(**params["input"])

        # model
        model_pars = params["model"]
        M = model_pars.pop("M", 20)
        dropout = model_pars.pop("dropout", 0.2)
        activation = model_pars.pop("activation", "relu")
        mod = tf.keras.models.Sequential(**model_pars)

        # compile
        mod.compile(**params["compile"])

        # fit
        fit = mod.fit(*train, validation_data=valid, **params["fit"])
        for k, v in fit.history.items():
            results[k].append(v)

    return results


# endregion

# %%
# region [1A]

# augment_frac = [0.0, 0.5, 1.0]
# augment_std = [0.05, 0.10]

# N = 1000 * 2 ** np.arange(5)
# train_frac = [0.6, 0.7, 0.8, 0.9]
params = [
    {
        "samples": 50,
        "input": {
            "N": N,
            "train_frac": train_frac,
            "augment_frac": 0.0,
            "augment_std": 0.0,
            "box": 100,
        },
        "model": {
            "name": "model",
            "layers": [
                tf.keras.layers.Dense(2, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
        },
        "compile": {
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"],
            "steps_per_execution": 4,
        },
        "fit": {"epochs": 500, "batch_size": 50, "verbose": 0},
    }
    for N in 1000 * 2 ** np.arange(5)
    for train_frac in [0.6, 0.7, 0.8, 0.9]
]

for p in params:
    res = train(p)

# endregion
