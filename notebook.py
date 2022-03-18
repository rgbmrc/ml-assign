# %%
# region defs

from collections import defaultdict
from itertools import product

import numpy as np
import tensorflow as tf


def generate(
    N, train_frac=1.0, rescale=None, offset=None, augment_frac=0.0, augment_std=0.0
):
    """
    This function generates the input data-set of N 2-dimensional points (x1,x2) in a square box and assoaciate them to a label classifying points inside/outside a triangle.
    It defines the fraction of data devoted to the training of the network and eventually the augemtations of such data with noise normally distributed.
    Args:
        N (int):
            Number of generated points
        train_frac (float, optional):
            Fraction of input data devoted to the network training. Defaults to 1.0.
        rescale (float, optional):
            Scaling factor applied to input data (x1,x2). Defaults to None.
        offset (float, optional):
            Offset applied to input data (x1,x2) after rescaling. Defaults to None.
        augment_frac (float, optional):
            Fraction of training data distorted with Gaussian noise without changing labels. Defaults to 0.0.
        augment_std (float, optional):
            std deviation of the Gaussian noise used for augmentation. Defaults to 0.0.
    Returns:
        (ndarray, ndarray) :
            2D input data points (x1,x2) and the associated true 1D labels (y)

    """
    x = np.random.random((N, 2)) * 100 - 50
    y = (x[..., 0] > -20) & (x[..., 1] > -40) & ((x[..., 0] + x[..., 1]) < 40)

    if rescale is None:
        rescale = x.mean()
    if offset is None:
        offset = x.std()
    x = (x - offset) / rescale

    N_train = int(N * train_frac)
    N_augment = int(N_train * augment_frac)
    x_train = np.pad(x[:N_train], ((0, N_augment), (0, 0)), mode="wrap")
    y_train = np.pad(y[:N_train], ((0, N_augment)), mode="wrap")
    x_train[:N_augment] += np.random.normal(0.0, augment_std, size=(N_augment, 2))
    valid = [a[N_train:] for a in (x, y)]
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


# **********************************************************************
# endregion

# %%
# region [1A]: INPUT SIZE N VS TRAINING FRACTION

# Dictionary of varying parameters
vals = {
    "N": 1000 * 2 ** np.arange(5),
    "train_frac": [0.6, 0.7, 0.8, 0.9],
}

# Dictionary of DNN configuration setups
params = [
    {
        "samples": 2,
        "input": {
            "N": N,
            "rescale": 50,
            "offset": 0,
            "train_frac": train_frac,
            "augment_frac": 0.0,
            "augment_std": 0.0,
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

results = []
for p in params:
    res = train(p)
    results.append(res["val_accuracy"])


data = np.asarray(results)
data = data[:, :, -1]
avg_data = np.mean(data, axis=1).reshape(4, 5)
# **********************************************************************
# endregion
# %%
# region [1B]: TRAINING FRACTION VS AUGMENTING FRACTION

# Dictionary of varying parameters
vals = {
    "train_frac": np.arange(6, 10) / 10,
    "augment_frac": np.arange(4) / 3,
}

# Dictionary of DNN configuration setups
params = [
    {
        "samples": 1,
        "input": {
            "N": 4000,
            "rescale": 50,
            "offset": 0,
            "train_frac": train_frac,
            "augment_frac": augment_frac,
            "augment_std": 0.1,
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
    for train_frac, augment_frac in product(*vals.values())
]

results = []
for p in params:
    res = train(p)
    results.append(res["val_accuracy"])

data = np.asarray(results)
data = data[:, :, -1]
avg_data = np.mean(data, axis=1).reshape(4, 5)

# **********************************************************************
# endregion
# %%
# region [2A]: ACTIVATION FUNCTIONS VS OPTIMIZER ALGORITHMS

# Dictionary of varying parameters
vals = {
    "optimization": ["sgd", "adamax", "rmsprop", "adam"],
    "activation": ["sigmoid", "relu", "tanh", "softsign", "elu"],
}


# Dictionary of DNN configuration setups
params = [
    {
        "samples": 1,
        "input": {
            "N": 8000,
            "rescale": 50,
            "offset": 0,
            "train_frac": 0.8,
            "augment_frac": 0.0,
        },
        "model": {
            "name": "model",
            "layers": [
                tf.keras.layers.Dense(2, activation=activation),
                tf.keras.layers.Dense(20, activation=activation),
                tf.keras.layers.Dense(20, activation=activation),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
        },
        "compile": {
            "loss": "binary_crossentropy",
            "optimizer": optimization,
            "metrics": ["accuracy"],
            "steps_per_execution": 4,
        },
        "fit": {"epochs": 500, "batch_size": 50, "verbose": 0},
    }
    for optimization, activation in product(*vals.values())
]

results = []
for p in params:
    res = train(p)
    results.append(res["val_accuracy"])

data = np.asarray(results)
data = data[:, :, -1]
avg_data = np.mean(data, axis=1).reshape(4, 5)

# **********************************************************************
# endregion
# %%
# region [2B]: INPUT SIZE N VS NUMBER OF HIDDEN NEURONS PER LAYER

# Dictionary of varying parameters
vals = {
    "N": 1000 * 2 ** np.arange(1, 5),
    # M is the number of neurons in each of the 2 Hidden Layers
    "M": np.arange(5, 26, 5),
}

# Dictionary of DNN configuration setups
params = [
    {
        "samples": 1,
        "input": {
            "N": N,
            "rescale": 50,
            "offset": 0,
            "train_frac": 0.8,
            "augment_frac": 0.0,
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
    for N, M in product(*vals.values())
]

results = []
for p in params:
    res = train(p)
    results.append(res["val_accuracy"])


data = np.asarray(results)
data = data[:, :, -1]
avg_data = np.mean(data, axis=1).reshape(4, 5)

# **********************************************************************
# endregion
# %%
# region [2C]: # HIDDEN LAYERS VS DROPOUT PERCENTAGE


def gen_arch(neurons, p_drop):
    """
    This function generates the list of layers for the DNN model architecture starting from a list of neurons per layer and the dropout probabiblity per layer
    Args:
        neurons (list):
            list of neurons per layer
        p_drop (float):
            dropout fraction applied to each hidden layer of the architecture

    Returns:
        list:
            list of layers for the DNN architecture
    """
    layers = [tf.keras.layers.Dense(2, activation="elu")]
    for n in neurons:
        layers.append(tf.keras.layers.Dense(n, activation="elu"))
        layers.append(tf.keras.layers.Dropout(p_drop))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    return layers


# Dictionary of varying parameters used to plot the grid search:
vals = {
    "K": np.arange(1, 6),
    "dropout": np.arange(4) / 10,
}


# Dictionary of DNN configuration setups
params = [
    {
        "samples": 1,
        "input": {
            "N": 4000,
            "rescale": 50,
            "offset": 0,
            "train_frac": 0.8,
            "augment_frac": 0.0,
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
    for p_drop in [0.0, 0.1, 0.2, 0.3]
    for neurons in [
        [125],
        [20, 20],
        [16, 16, 10],
        [11, 12, 12, 12],
        [11, 10, 10, 10, 10],
    ]
]

results = []
for p in params:
    res = train(p)
    results.append(res["val_accuracy"])


data = np.asarray(results)
data = data[:, :, -1]
avg_data = np.mean(data, axis=1).reshape(4, 5)

# **********************************************************************
# endregion
