# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14

# %% [markdown]
# ### Import Libraries and define functions to build and train a XGBoost Model

# %%
# USE tfresh to extract features
from tsfresh import extract_features

# XGBoost
from xgboost import XGBClassifier

# Get the final accuracy from the trained XBGboost Model
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## Generate and preprocess TIME_SERIES data
# Define a unique function that generates 1D data (time series) for both the analyses (via CNN \& via XGBoost).
#
# Rather, the preprocessing is called by two different functions (one for CNN and one for XGBoost), according to the protocols.
#
# In particular, as for the XGBoost protocol, we
#
# 1) rewrite the input data $x$ as a PANDAS data-frame;
#
# 2) extract all the possible features of data via TSFRESH (XGBoost should work even before rescaling);
#
# 3) perform the data-rescaling (remove data-average and standardize data).

# %%
def generate_1D_data(N, L, drift, noise_amp, pattern_width, pattern_amp):
    rng = np.random.default_rng(12345)

    x = rng.normal(drift, noise_amp, N * L).cumsum().reshape(N, L).astype(int)
    pattern = (pattern_amp * np.sin(np.linspace(0, np.pi, pattern_width))).astype(int)
    y = np.arange(N) % 3
    for i, t0 in enumerate(rng.integers(0, L - 1 - pattern_width, N)):
        x[i, t0 : t0 + pattern.size] += (y[i] - 1) * pattern

    return x, y


def generate_1D_data_from_folder():
    str0 = "ts_L60_Z12_A500_DX50_bias5_N10000.dat"
    fnamex = "DATA/x_" + str0
    fnamey = "DATA/y_" + str0

    data_x = np.loadtxt(fnamex, delimiter=" ", dtype=float)
    N, L = len(data_x), len(data_x[0])
    # note: here it does not need to be converted to the 3-bit version, a label remains y[i]=0,1,2
    data_y = np.loadtxt(fnamey, dtype=int)

    return data_x, data_y


def CNN_preprocess_1D_data(x, z, train_frac):
    N = z.size
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


def get_dataframe(x):
    """
    Build the input for XGBoost Classifier as DATAFRAME.
    For more: https://tsfresh.readthedocs.io/en/latest/text/data_formats.html

    It results into a dictionary of 3 values:
    -first column: id of the sample
    -second column: id of the time step
    -third column: value of the feature of _id_-sample at _id_-step

    Args:
        x (ndarray): array of time series, (#samples,time,1)
    Returns:
        Pandas.DataFrame: A dataframe ready for features extraction
    """
    # N = # samples
    # t = timesteps
    N, t = x.shape[0], x.shape[1]
    # build id columns
    id_col = np.repeat(np.arange(N), t)
    # build time columns
    time_col = np.tile(np.arange(t), N)
    # build var columns
    x_col = x.flatten()

    # BUILD A DICTIONARY FOR THE DATAFRAME
    x_dict = {"id": id_col, "time": time_col, "value": x_col}

    return pd.DataFrame(x_dict)


def XGB_preprocess_1D_data(data_x, data_y, train_frac):
    # 1) BUILD DATAFRAME
    data_frame = get_dataframe(data_x)
    print(data_frame)

    # 2) EXTRACT FEATURES
    x_features = extract_features(
        data_frame,  # our dataframe
        column_id="id",  # sample id, from 0 to N
        column_sort="time",  # timestep, from 0 to t
        column_kind=None,  # we have only one feature
        column_value="value",  # value of input
        n_jobs=8,
    )  # number of cores
    print(x_features)

    # 3) REMOVE COLUMNS with NaN or inf
    x_features.replace([np.inf, -np.inf], np.nan)
    x_features = x_features.dropna(axis="columns")
    print(x_features)

    # 4) GET TRAINING AND VALIDATION
    N = data_y.size
    N_train = int(train_frac * N)
    x_train = x_features[:N_train]
    y_train = data_y[:N_train]
    x_val = x_features[N_train:]
    y_val = data_y[N_train:]

    # 5) RESCALE: in each feature, remove average and divide by std
    average = np.mean(x_train, axis=0)
    x_train -= average
    x_val -= average

    std = np.std(x_train, axis=0)
    x_train /= std
    x_val /= std

    return x_train, y_train, x_val, y_val


# %% [markdown]
# ### Define input-parameters for the CNN/XGBoost training

# %%
default_params = {
    # Data-generation (used for both CNN & XGBoost)
    "generate": {
        "N": 10000,
        "L": 60,
        "drift": 5.0,
        "noise_amp": 50.0,
        "pattern_amp": 500.0,
        "pattern_width": 12,
    },
    # Data-Preprocessing (used for both CNN & XGBoost)
    "preprocess": {"train_frac": 0.8},
    # CNN MODEL
    "CNN_model": {"N_filters": 5, "reg_strength": {"l1": 0.0, "l2": 0.0}},
    "CNN_fit": {"epochs": 200, "batch_size": 250, "shuffle": True, "verbose": 0},
    # XGBoost MODEL
    "XGB_model": {
        "max_depth": 6,
        "min_child_weight": 1,  # minimim number of data at the last level of the Tree
        "learning_rate": 0.3,
        "use_label_encoder": False,
    },
}

# %% [markdown]
# # Generate Data

# %%
# GENERATE DATA
data_x, data_y = generate_1D_data(**default_params["generate"])
# data_x, data_y = generate_1D_data_from_folder()
# PREPROCESS data for XGBoost
x_train, y_train, x_val, y_val = XGB_preprocess_1D_data(
    data_x, data_y, **default_params["preprocess"]
)

# %% [markdown]
# ## XGBoost Analysis

# %%
# reproducibility
np.random.seed(12345)

# define parameters for xgboost
"""params = {
    'max_depth':6,
    'min_child_weight':1,   # minimim number of data at the last level of the Tree
    'learning_rate':0.3,
    'use_label_encoder':False
    }"""

# build model with given params
model = XGBClassifier(**default_params["XGB_model"])
# fit the model
model.fit(x_train.values, y_train)

# predict labels on training set
y_pred_train = model.predict(x_train)
# predict labels on validation set
y_pred_val = model.predict(x_val)

y_pred_val_soft = model.predict_proba(x_val)

# compute accuracies
acc_train = accuracy_score(y_train, y_pred_train)
acc_val = accuracy_score(y_val, y_pred_val)

# print accuracies
print("Training accuracy:", acc_train)
print("Validation accuracy:", acc_val)

# %% [markdown]
# ### Import Libraries and define functions to build and train a CNN

# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

# Generate CNN model
def CNN_build_model(L, N_classes, reg_strength, N_filters):
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
    (x_train, y_train), valid = CNN_preprocess_1D_data(*data, **params["preprocess"])
    model = CNN_build_model(x_train.shape[1], y_train.shape[1], **params["model"])
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
# # XGBoost Analysis

# %%
def XGB_train_model(default_params):
    # reproducibility
    np.random.seed(12345)
    # Generate Data
    data_x, data_y = generate_1D_data(**default_params["generate"])
    # data_x, data_y = generate_1D_data_from_folder()
    # Preprocess Data
    x_train, y_train, x_val, y_val = XGB_preprocess_1D_data(
        data_x, data_y, **default_params["preprocess"]
    )
    # Define Model
    model = XGBClassifier(**default_params["XGB_model"])
    # Train Model
    model.fit(x_train.values, y_train)
    # predict labels on training set
    y_pred_train = model.predict(x_train)
    # predict labels on validation set
    y_pred_val = model.predict(x_val)
    # compute accuracies
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)
    # print accuracies
    print("Training accuracy:", acc_train)
    print("Validation accuracy:", acc_val)
    return acc_train, acc_val


def get_important_features(x_features):
    import seaborn as sns

    # get feature names
    feature_names = x_features.columns.values
    print(feature_names.shape)
    # empty dict
    feat_imp = {}
    # get f importance from model
    for ii, f in enumerate(feature_names):
        feat_imp[f] = model.feature_importances_[ii]
    # sort features depending on their importances
    feat_imp = dict(sorted(feat_imp.items(), reverse=True, key=lambda item: item[1]))

    minval = 0.005
    print(
        "------------- Feature importance sum = "
        + str(np.sum(model.feature_importances_))
    )
    print("------------- Showing feature with importance > " + str(minval))
    for ii, k in enumerate(feat_imp.keys()):
        if feat_imp[k] > minval:
            print(ii + 1, k, feat_imp[k])

    # get feature names given importance order
    features = list(feat_imp.keys())

    # build dataframe for sns pairplot
    df = pd.DataFrame(
        {
            "1": x_features[features[0]],
            "2": x_features[features[1]],
            "3": x_features[features[2]],
            "4": x_features[features[3]],
            "5": x_features[features[4]],
            "Class": data_y,
        }
    )
    # pairplot with seaborn
    n_class = 3
    pal = sns.blend_palette(["blue", "red", "gold"], n_class)
    sns.pairplot(df, hue="Class", plot_kws=dict(alpha=0.8), palette=pal)


# %%
