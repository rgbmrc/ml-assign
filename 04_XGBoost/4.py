# %% [markdown]
# # LCPB 21-22 Exercise 4, XGBoost - Group 2205

# %%
# %% [markdown]
# ### Imports, settings & helper functions
import warnings

import numpy as np
import pandas as pd
import IPython.display
import matplotlib.pyplot as plt
import seaborn as sns

# ML specific libraries
import tensorflow as tf  # (convolutinal) neural network
import xgboost as xgb  # gradient boosted decision tree
import tsfresh  # extract features

warnings.simplefilter("ignore")
plt.rcParams["font.size"] = 12
IPython.display.set_matplotlib_formats("svg")  # vector graphics


def split_arrays(fracs, *arrays):
    """
    Splits one or more arrays at the given fractions along the 0th axis.
    """
    return zip(*(np.split(a, [round(f * len(a)) for f in fracs]) for a in arrays))


# %% [markdown]
# ## Part 1: time series

# %% [markdown]
# ### Common parameters for the trainings

generate_params = {
    "L": 60,
    "drift": 5.0,
    "noise_amp": 50.0,
    "pattern_amp": 500.0,
    "pattern_width": 12,
}
train_params = {
    # CNN
    "CNN": {
        "model": {"N_filters": 5, "reg_strength": {"l1": 0.0, "l2": 0.0}},
        "fit": {"epochs": 200, "batch_size": 250, "shuffle": True, "verbose": 0},
    },
    # XGBoost
    "XGB": {
        "model": {
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.3,
            "use_label_encoder": False,
            "verbosity": 0,
        },
        "fit": {},
    },
}

# %% [markdown]
# ### Functions (model specific ones are prefixed by CNN/XGB)

# %% [markdown]
# #### Generating the data
#
# Signal labels are assigned in the following way:
# 0 = negative bump,
# 1 = no bump,
# 2 = positive bump.


def generate_1D_data(N, L, drift, noise_amp, pattern_width, pattern_amp, seed=None):
    rng = np.random.default_rng(seed)  # reproducibility: fix random seed

    x = rng.normal(drift, noise_amp, N * L).cumsum().reshape(N, L).astype(int)
    pattern = (pattern_amp * np.sin(np.linspace(0, np.pi, pattern_width))).astype(int)
    y = np.arange(N) % 3
    for i, t0 in enumerate(rng.integers(0, L - 1 - pattern_width, N)):
        x[i, t0 : t0 + pattern.size] += (y[i] - 1) * pattern

    return x, y


# %% [markdown]
# #### Preprocessing the data
#
# In the CNN case, the preprocessing is the same of that in Excercise 3.
#
# For XGBoost, we
# 1) Recast the input data as a `pandas` dataframe
# 2) Extract some features via `tsfresh`
# 3) For each feature, subtract the mean and rescale by the sdt. dev.


def CNN_preprocess_1D_data(x, y):
    N = y.size
    # remove mean of each signal & rescale by overall std
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (2.0 * np.std(x))

    # quantized labels
    lbls, ilbls = np.unique(y, return_inverse=True)
    y = np.zeros((N, lbls.size))
    y[np.arange(N), ilbls] = 1.0
    x = x[..., np.newaxis]

    return x, y


def XGB_preprocess_1D_data(x, y):
    N, T = x.shape  # samples & timesteps
    # dataframe
    x = pd.DataFrame(
        {
            "i": np.repeat(np.arange(N), T),
            "t": np.tile(np.arange(T), N),
            "x": x.flatten(),
        }
    )
    # extract features
    x = tsfresh.extract_features(
        x,
        column_id="i",  # sample id, from 0 to N
        column_sort="t",  # timestep, from 0 to t
        column_value="x",  # i-th signal value at time t
        n_jobs=8,  # number of cores
        disable_progressbar=True,
    )
    # rescale
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    # remove columns with nan or inf (inplace)
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.dropna(axis="columns", inplace=True)

    return x, y


# %% [markdown]
# #### Building the models
#
# The CNN is the one introduced in Excercise 3.


def CNN_build_model(x, y, reg_strength, N_filters):
    tf.random.set_seed(12345)
    reg = tf.keras.regularizers.l1_l2(**reg_strength)
    ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv1D(
            filters=N_filters,
            kernel_size=11,
            kernel_initializer=ini,
            kernel_regularizer=reg,
            activation="relu",
            input_shape=(x.shape[1], 1),
        )
    )
    model.add(tf.keras.layers.AveragePooling1D(5))  # model.add(MaxPooling1D(3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(12, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(y.shape[1], activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def XGB_build_model(x, y, **params):
    return xgb.XGBClassifier(**params)


# %% [markdown]
# #### Assesing the models accuracy


def CNN_compute_accuracy(model, *data):
    return model.evaluate(*data)[model.metrics_names.index("accuracy")]


def XGB_compute_accuracy(model, *data):
    return model.score(*data)


# %% [markdown]
# ## Part 1a - CNN vs XGB for small datasets
#
# The performances of XGB (gradient boosted decision tree) and CNN (convolutinal neural network)
# are compared for small datasets.
#
# For each dataset size `N`, we repeat the training multiple times in order to collect some statistics.
# The overall number of samples (population size, `pop`) is fixed.

# %%
# %%capture --no-display
# uncomment previous line to suppress output

pop = 500
N_list = [20, 50, 100, 150, 200, 250, 300, 400, 500]

accuracy = np.ma.masked_all((len(train_params), len(N_list), pop // min(N_list)))
for j, N in enumerate(N_list):
    for k in range(pop // N):
        data = generate_1D_data(N=N, **generate_params)
        for i, (typ, params) in enumerate(train_params.items()):
            # retrieve the module level function "typ_fname"
            get_func = lambda fname: globals()[f"{typ}_{fname}"]
            # preprocess data and split training and validation sets
            inputs = get_func("preprocess_1D_data")(*data)
            train, valid = split_arrays([0.8], *inputs)
            # build and train the model
            model = get_func("build_model")(*train, **params["model"])
            model.fit(*train, **params["fit"])
            # compute accuracy
            accuracy[i, j, k] = get_func("compute_accuracy")(model, *valid)

# plot
for typ, acc in zip(train_params, accuracy):
    plt.errorbar(N_list, acc.mean(1), acc.std(1), capsize=4, capthick=1.5, label=typ)
    plt.ylabel("validation accuracy")
    plt.xlabel("dataset size $N$")
plt.xlim(left=0)
plt.ylim(top=1)
plt.grid()
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# For small datasets, XGB performs better than CNN in recognizing the studied singal's patterns.
# It is both more accurate and more robust (less fluctuations). CNN only starts to catch up at N > ~500.

# %% [markdown]
# ## Part 1b - feature interpretation

# %%
# generate large sample
data = generate_1D_data(N=1000, seed=1, **generate_params)
train = XGB_preprocess_1D_data(*data)
model = XGB_build_model(*train, **train_params["XGB"]["model"])
model.fit(*train, **train_params["XGB"]["fit"])

# dict of sorted features
x, y = train
srt = np.argsort(model.feature_importances_)[::-1]
features = dict(zip(x.columns.values[srt], model.feature_importances_[srt]))

thresh = 0.01
print(f"Features with importance > {thresh}")
for i, (feat, imp) in enumerate(features.items()):
    if imp < thresh:
        break
    print(i + 1, feat, imp)

# sns pairplot
num = 4
xy = pd.DataFrame({i + 1: x[feat] for i, feat in zip(range(num), features)} | {"y": y})
sns.pairplot(xy, hue="y", palette=sns.color_palette("husl", 3))
plt.show()


# %% [markdown]
# The most relevant features for classification appear to be of type (0 <= L < H <= 1)
#
# `x__change_quantiles__f_agg_"var"__isabs_False__qh_`H`__ql_`L
#
# These features are computed by:
# - Selecting the slice of the signal where it's value is enclosed between the L and the H quatiles
# - Evaluating the differences of the signal values at consecutive timestamps
# - Computing the variance of said differences
#
# Signals with labels 0 and 2 are expected to have larger variances in the
# lower and upper part of their spectrum respectively. Inded, as shown by the plot,
# - L ~ 0, H ~ 1 (feature 3), signle out signals of class 1 (lower variance)
# - H < ~0.5 (features 2, 4),  signle out signals of class 0 (higher variance)
# - L > ~0.5 (feature 1),  signle out signals of class 2 (higher variance)

# %%
