# %%
# region defs

from collections import defaultdict
from itertools import product

import numpy as np
import tensorflow as tf


def generate(N, box=1, train_frac=1.0, augment_frac=0.0, augment_std=0.1):
    """
    This function generates the input data-set of N 2-dimensional points (x1,x2) in a square box and assoaciete them a label classifying points inside/outside a triangle.
    It defines the fraction of data devoted to the training of the network and eventually the augemtations of such data with noise normally distributed.
    Args:
        N (int):
            Number of generated points
        box (int, optional):
            Size of the square box of data-points. Defaults to 1.
        train_frac (float, optional):
            Fraction of input data devoted to the network training. Defaults to 1.0.
        augment_frac (float, optional):
            Fraction of training data distorted with Gaussian noise without changing labels. Defaults to 0.0.
        augment_std (float, optional):
            std deviation of the Gaussian noise used for augmentation. Defaults to 0.1.
    Returns:
        ndarray, ndarray :
            2D input data points and the associated 1D labels
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
    """
    This function trains multiple times a Deep Neural Network with a chosen setup given by the parameters in 'params' and save the results of each trial into a dictionary.

    Args:
        params (dictionary of dictionaries): _description_

    Returns:
       dictionary : _description_
    """
    samples = params.setdefault("samples", 1)
    results = defaultdict(list)

    for _ in range(samples):
        # input
        train, valid = generate(**params["input"])

        # model
        mod = tf.keras.models.Sequential(**params["model"])

        # compile
        mod.compile(**params["compile"])

        # fit
        fit = mod.fit(*train, validation_data=valid, **params["fit"])
        for k, v in fit.history.items():
            results[k].append(v)

    return results


# endregion

# %%
# region [1A]: INPUT SIZE N VS TRAINING FRACTION

vals = {
    "N": 1000 * 2 ** np.arange(5),
    "train_frac": [0.6, 0.7, 0.8, 0.9],
}
params = [
    {
        "samples": 1,
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
    for N, train_frac in product(*vals.values())
]

for p in params:
    res = train(p)

# endregion
# %%
# region [1B]: TRAINING FRACTION VS AUGMENTIN FRACTION

params = [
    {
        "samples": 1,
        "input": {
            "N": 4000,
            "train_frac": train_frac,
            "augment_frac": augment_frac,
            "augment_std": 0.1,
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
    for train_frac in [0.6, 0.7, 0.8, 0.9]
    for augment_frac in [0.0, 0.5, 1.0]
]

for p in params:
    res = train(p)

# endregion
# %%
# region [2A]: ACTIVATION FUNCTIONS VS OPTIMIZER ALGORITHMS

params = [
    {
        "samples": 1,
        "input": {
            "N": 8000,
            "train_frac": 0.8,
            "augment_frac": 0.0,
            "box": 100,
        },
        "model": {
            "name": "model",
            "layers": [
                tf.keras.layers.Dense(2, activation=act_function),
                tf.keras.layers.Dense(20, activation=act_function),
                tf.keras.layers.Dense(20, activation=act_function),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
        },
        "compile": {
            "loss": "binary_crossentropy",
            "optimizer": opt_algorithm,
            "metrics": ["accuracy"],
            "steps_per_execution": 4,
        },
        "fit": {"epochs": 500, "batch_size": 50, "verbose": 0},
    }
    for opt_algorithm in ["sgd", "adamax", "rmsprop", "adam"]
    for act_function in ["sigmoid", "relu", "tanh", "softsign", "elu"]
]

for p in params:
    res = train(p)

# endregion
# %%
# region [2B]: INPUT SIZE N VS NUMBER OF HIDDEN NEURONS PER LAYER

params = [
    {
        "samples": 1,
        "input": {
            "N": N,
            "train_frac": 0.8,
            "augment_frac": 0.0,
            "box": 100,
        },
        "model": {
            "name": "model",
            "layers": [
                tf.keras.layers.Dense(2, activation="elu"),
                tf.keras.layers.Dense(M, activation="elu"),
                tf.keras.layers.Dense(M, activation="elu"),
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
    for M in [5, 10, 15, 20, 25]
]

for p in params:
    res = train(p)

# endregion
# %%
# region [2C]: # HIDDEN LAYERS VS DROPOUT PERCENTAGE


def gen_arch(neurons, p_drop):
    layers = [tf.keras.layers.Dense(2, activation="elu")]
    for n in neurons:
        layers.append(tf.keras.layers.Dense(n, activation="elu"))
        layers.append(tf.keras.layers.Dropout(p_drop))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    return layers


params = [
    {
        "samples": 1,
        "input": {
            "N": 4000,
            "train_frac": 0.8,
            "augment_frac": 0.0,
            "box": 100,
        },
        "model": {
            "name": "model",
            "layers": gen_arch(neurons, p_drop),
        },
        "compile": {
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"],
            "steps_per_execution": 4,
        },
        "fit": {"epochs": 500, "batch_size": 50, "verbose": 0},
    }
    for p_drop in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for neurons in [
        [125],
        [20, 20],
        [16, 16, 10],
        [11, 12, 12, 12],
        [11, 10, 10, 10, 10],
    ]
]

for p in params:
    res = train(p)

# endregion
# %%
