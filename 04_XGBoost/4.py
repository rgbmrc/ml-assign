# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14

# %% [markdown]
# ### Import Libraries and define functions to build and train a XGBoost Model

# %%
# IMPORT THE 2 CLASSIFIERS we can use
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# USE tfresh to extract features
from tsfresh import extract_features

# XGBoost
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance, to_graphviz, plot_tree

print("XGBoost version:", xgboost.__version__)


# %% [markdown]
# ### Import Libraries and define functions to build and train a CNN

# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

# Generate CNN model
def build_model(L, N_classes, reg_strength, N_filters):
    tf.random.set_seed(12345)
    reg = tf.keras.regularizers.l1_l2(**reg_strength)
    ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

    model = Sequential()
    model.add(
        Conv1D(
            filters=N_filters,
            kernel_size=11,
            kernel_initializer=ini,
            kernel_regularizer=reg,
            activation="relu",
            input_shape=(L, 1),
        )
    )
    model.add(AveragePooling1D(5))  # model.add(MaxPooling1D(3))
    model.add(Conv1D(filters=5, kernel_size=7, activation="relu"))
    model.add(Flatten())
    model.add(Dense(12, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(N_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def CNN_train_model(params):
    data = generate_1D_data(**params["generate"])
    (x_train, y_train), valid = preprocess_1D_data(*data, **params["preprocess"])
    model = build_model(x_train.shape[1], y_train.shape[1], **params["model"])
    model.fit(x_train, y_train, validation_data=valid, **params["fit"])
    return (x_train, y_train), valid


# Plot of training history
def show_CNN_history(fit):
    fig, AX = plt.subplots(1, 2, figsize=(12, 5.0))
    ax = AX[0]
    ax.plot(fit.history["accuracy"], "b", label="train")
    ax.plot(fit.history["val_accuracy"], "r--", label="valid.")
    ax.axhline(1 / 3, ":", c="gray", label="random choice")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    ax.legend()
    ax = AX[1]
    ax.plot(fit.history["loss"], "b", label="train")
    ax.plot(fit.history["val_loss"], "r--", label="valid.")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim([0, 1.05 * np.max(fit.history["loss"])])
    ax.legend()
    plt.show()


# %% [markdown]
# ## Generate and preprocess TIME_SERIES data as in Exercise 3
# We know that the average value of a sample is not relevant: let's see if XGBoost works if we do not remove such average and we do not standardize data

# %%
def generate_1D_data(N, L, drift, noise_amp, pattern_width, pattern_amp):
    rng = np.random.default_rng(12345)

    x = rng.normal(drift, noise_amp, N * L).cumsum().reshape(N, L).astype(int)
    pattern = (pattern_amp * np.sin(np.linspace(0, np.pi, pattern_width))).astype(int)
    y = np.arange(N) % 3
    for i, t0 in enumerate(rng.integers(0, L - 1 - pattern_width, N)):
        x[i, t0 : t0 + pattern.size] += (y[i] - 1) * pattern

    return x, y


def preprocess_1D_data(x, z, train_frac, rescale=False):
    N = z.size
    if rescale:
        # remove mean of each signal & rescale by overall std
        x = x - x.mean(axis=1, keepdims=True)
        x = x / (2.0 * np.std(x))

    # quantized labels
    lbls, ilbls = np.unique(z, return_inverse=True)
    y = np.zeros((N, lbls.size))
    y[np.arange(N), ilbls] = 1.0
    x = x[..., np.newaxis]

    # split train and validation
    xs, ys = (np.split(a, [int(train_frac * N)]) for a in (x, y))
    return zip(xs, ys)


# %% [markdown]
# ### Define input parameters for the CNN training

# %%
default_params = {
    "generate": {
        "N": 10000,
        "L": 60,
        "drift": 5.0,
        "noise_amp": 50.0,
        "pattern_amp": 500.0,
        "pattern_width": 12,
    },
    "preprocess": {"train_frac": 0.8},
    "model": {"N_filters": 5, "reg_strength": {"l1": 0.0, "l2": 0.0}},
    "fit": {"epochs": 200, "batch_size": 250, "shuffle": True, "verbose": 0},
}

# %% [markdown]
# Build the input for XGBoost Classifier as DATAFRAME.
#
# It results into a dictionary of 3 values:
#
# first column $\to$ id of the sample
#
# second column $\to$  id of the time step
#
# third column $\to$ value of the feature of _id_-sample at _id_-step
#
# For more: https://tsfresh.readthedocs.io/en/latest/text/data_formats.html

# %%
def get_dataframe(x):
    """Build input dataframe for given data series
    Input:
    var = array of time series, (#samples,time,1)
    Return:
    df = dataframe ready for features extraction
    """

    # N = #samples, t = timesteps
    N, t = x.shape[0], x.shape[1]
    # build id columns
    id_col = np.repeat(np.arange(N), t)
    # build time columns
    time_col = np.tile(np.arange(t), N)
    # build var columns
    x_col = x.flatten()

    # build dict for df
    x_dict = {"id": id_col, "time": time_col, "value": x_col}

    # return dataframe
    return pd.DataFrame(x_dict)


# %%
# extract features

data = generate_1D_data(**default_params["generate"])
(x_train, y_train), valid = preprocess_1D_data(*data, **default_params["preprocess"])
dataframe = get_dataframe(x_train)
x_features = extract_features(
    dataframe,  # our dataframe
    column_id="id",  # sample id, from 0 to N
    column_sort="time",  # timestep, from 0 to t
    column_kind=None,  # we have only one feature
    column_value="value",  # value of input
    n_jobs=4,
)  # number of cores

# %%
# region classify


def classify(clf=GradientBoostingClassifier(), show=False):
    # GradientBoostingClassifier():
    #   n_estimators = 100 (default)
    #   loss function = deviance(default) used in Logistic Regression
    # XGBClassifier()
    #   n_estimators = 100 (default)
    #   max_depth = 3 (default)
    clf.fit(x_train, y_train)
    # predictions
    y_hat = clf.predict(x_test)

    if CASE < 10:
        print("errors: {:.2f}%".format(100 * (1 - clf.score(x_test, y_test))))

    dx = 0.02
    x_seq = np.arange(-S, S + dx, dx)
    nx = len(x_seq)
    x_plot = np.zeros((nx * nx, 2))
    q = 0
    for i in range(nx):
        for j in range(nx):
            x_plot[q, :] = [x_seq[i], x_seq[j]]
            q += 1
    y_plot = clf.predict(x_plot)

    plt.figure(figsize=(6, 6))
    plt.title(str(clf))

    scat(x_plot, y_plot, cmap="winter", s=1)
    scat(x_train, y_train, s=7)
    plt.show()

    if show:
        dump_list = clf.get_booster().get_dump()
        num_trees = len(dump_list)
        print("num_trees=", num_trees)

        fig, AX = plt.subplots(3, 1, figsize=(30, 30))
        for i in range(min(3, num_trees)):
            ax = AX[i]
            plot_tree(clf, num_trees=i, ax=ax)
        fig.savefig("DATA/tree-classif.png", dpi=300, pad_inches=0.02)
        plt.show()


# %%
classify(
    XGBClassifier(
        seed=1,
        # objective='binary:logistic',
        objective="reg:squarederror",
        eval_metric="rmse",
        reg_lambda=0.001,
        n_estimators=10,
    ),
    show=True,
)

# endregion
