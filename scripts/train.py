import numpy as np
import tensorflow as tf

from simsio import run_sim
from simsio.serializers import NPZSerializer
from simsio.extensions.ext_keras import KerasModelSerializer, KerasWeightSerializer


def generate(
    N, train_frac=1.0, rescale=None, offset=None, augment_frac=0.0, augment_std=0.0
):
    """
    Data generation.
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


with run_sim() as sim:
    sim.link("model", serializer=KerasModelSerializer())
    samples = sim.par["samples"]

    for s in range(samples):

        # IO handles
        sim.link(f"weights_{s}", serializer=KerasWeightSerializer())
        sim.link(f"history_{s}", serializer=NPZSerializer())

        # input
        train, valid = generate(**sim.par["input"])

        # model
        model_pars = sim.par["model"].copy()
        model_vars = model_pars.pop("vars", {})
        model_pars["layers"] = [
            eval(l, globals(), model_vars) for l in model_pars["layers"]
        ]
        mod = tf.keras.models.Sequential(**model_pars)
        sim[f"weights_{s}"] = mod

        # compile
        compile_pars = sim.par["compile"].copy()
        compile_pars["optimizer"] = tf.keras.optimizers.get(compile_pars["optimizer"])
        mod.compile(**sim.par["compile"])

        # fit
        fit = mod.fit(*train, validation_data=valid, **sim.par["fit"])
        sim[f"history_{s}"] = fit.history

        # dump
        sim.dump()

    sim[f"model"] = mod
