# %%
import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import stats

# %%


def generate_data_point(PDF_name, PDF_params, Exp_params):
    # This function generate a histogram from a given PDF
    PDF = getattr(stats, f"{PDF_name}")
    PDF = getattr(PDF, "rvs")
    # Generate a lot of data points
    x = PDF(**PDF_params, size=10000)
    # print(x.shape)
    # Cut the array in between a finite range
    x = x[x > Exp_params["lower_bound"]]
    x = x[x < Exp_params["upper_bound"]]
    # print(x.shape)
    x = x[: Exp_params["n_points"]]
    # print(x.shape)
    x, bin_edges = np.histogram(x, bins=Exp_params["bins"], density=True)
    x /= np.sum(x)
    return x


PDF_list = [
    ("expon", {"loc": 0.0, "scale": 1.0}),
    ("norm", {"loc": 0.0, "scale": 1.0}),
    ("cauchy", {"loc": 0.0, "scale": 1.0}),
    ("pareto", {"b": 3, "loc": -1.0, "scale": 1.0}),
    # Power law. Since Pareto dist is defined for x>=1 and the selected sampling range is [0,1], it's translated by -1 (loc)
]

Exp_params = {
    "lower_bound": 0,
    "upper_bound": 50.0,
    "n_points": 1000,
    "bins": 50,
}

# %%
ndata = 1000
data = []
labels = []
for i in range(ndata):
    data.append(generate_data_point(*PDF_list[i % len(PDF_list)], Exp_params))
    labels.append(i % len(PDF_list))

x = np.asarray(data)
y = np.asarray(labels)


# %%
def visualize(x_reduced, y, ax=None):
    ax = ax or plt.axes()
    ax.scatter(*x_reduced.T, s=3, c=y)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


# %%
pair_dist = sp.spatial.distance.cdist(x, x, metric="jensenshannon")

TSNE_params = {
    "n_components": 2,
    "init": "random",  # ?
    "learning_rate": "auto",
    "random_state": 0,
    "metric": "precomputed",
    "square_distances": True,
}
perplexity = np.logspace(-0.25, 3, 14)

fig, axs = plt.subplots(2, 7, figsize=(22, 6))

for ax, p in zip(axs.flat, perplexity):
    ax.set_title(p)
    x_reduced = TSNE(perplexity=p, **TSNE_params).fit_transform(pair_dist)
    visualize(x_reduced, y, ax)


# %%
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# plt.scatter(*x_reduced.T, c=labels)

# %%

x_reduced = TSNE(perplexity=10, **TSNE_params).fit_transform(pair_dist)
k_dist = np.mean(np.sort(pair_dist, 1), 0)  # [10:-10]
# min_pts = np.argsort(np.diff(np.diff(k_dist)))[::-1]
# eps = k_dist[min_pts]
# print(eps[0])

# %%
fig, axs = plt.subplots(5, 11, figsize=(30, 15))

from itertools import product

for ax, (pts, eps) in zip(
    axs.flat, product(np.arange(1, 100, 20), np.linspace(0.08, 0.12, 11))
):

    DBSCAN_params = {
        "eps": eps,  # ?
        "min_samples": pts,  # ?
        "metric": "precomputed",
    }
    model = DBSCAN(**DBSCAN_params)
    model.fit(pair_dist)
    visualize(x_reduced, model.labels_, ax)
    ax.text(0, 0, f"{pts}, {eps:.3f}", transform=ax.transAxes)

# %%
