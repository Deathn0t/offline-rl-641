# %%

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import warnings
# warnings.filterwarnings('ignore')

import time

import functools

import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# %matplotlib notebook

# %%


def polynom_basis(coefs):
    degree = len(coefs) - 1
    return [lambda x: coefs[i] * np.cos(x)*np.power(x, i) for i in range(degree + 1)]


def random_polynom(degree: int = 2) -> callable:
    """Generate a random polynome function: x -> y"""
    coefs = np.random.uniform(low=-1.0, high=1.0, size=degree + 1)
    basis = polynom_basis(coefs)

    return coefs, basis


def polynom(x: float, basis) -> float:
    return functools.reduce(lambda cum, f: cum + f(x), basis, 0)


# %%


def generate_data_from_f(
    f: callable, x_min: float = -1, x_max: float = +1, n_samples: int = 100
) -> tuple:
    x = np.linspace(x_min, x_max, n_samples)
    y = np.array([f(xi) for xi in x])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return (X_train, y_train), (X_test, y_test)


# %%

# Example: how to use "random_polynom"


plt.figure()

for _ in range(10):
    degree = np.random.randint(1,4)
    coefs, basis = random_polynom(degree)
    poly0 = functools.partial(polynom, basis=basis)

    (x_train, y_train), (x_test, y_test) = generate_data_from_f(poly0, -1, 1)

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    plt.scatter(x, y, label=f"{degree}")

plt.legend()
plt.show()


# %%

# generate neural networks


def generate_nn(nunits=[10, 10], activation=["relu", "relu"]):

    assert len(nunits) == len(activation)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))

    for i, (nunits_, activation_) in enumerate(zip(nunits, activation)):
        model.add(
            tf.keras.layers.Dense(nunits_, activation=activation_, name=f"layer_{i+1}")
        )

    model.add(tf.keras.layers.Dense(1))

    return model


def generate_random_net(
    max_number_layer=10, max_number_units=10, activations=["tanh", "relu"]
):

    layers = np.random.randint(max_number_layer)

    nunits = []
    layer_activations = []

    for layer in range(layers):

        units = np.random.randint(low=1, high=max_number_units + 1)
        activation = np.random.choice(activations)
        nunits.append(units)
        layer_activations.append(activation)

    return nunits, layer_activations


def fit_nn(model, x_train, y_train):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
    )

    model.fit(x_train, y_train, epochs=100, batch_size=4, verbose=0)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test, batch_size=1)
    return r2_score(predictions, y_test)


# %%

t1 = time.time()

# generate data
n_func = 300
degree_max = 3
data = []
n_networks = 5

networks = dict()

# networks[0] = {'units':[10,10], 'activations': ["relu", "relu"]}

# generate random NNs
# for i in range(n_networks):
#     nunits, activations_ = generate_random_net(10, 10, activations=["tanh", "relu"])
#     networks[i] = {'units':nunits, 'activations': activations_}


networks = {
    0: {"units": [8, 10], "activations": ["tanh", "relu"]},
    1: {"units": [9, 5], "activations": ["relu", "tanh"]},
    2: {"units": [9, 3, 1, 3], "activations": ["tanh", "tanh", "relu", "tanh"]},
    3: {
        "units": [4, 4, 1, 10, 10, 9],
        "activations": ["relu", "tanh", "relu", "tanh", "tanh", "relu"],
    },
    4: {"units": [7], "activations": ["tanh"]},
}

for j in range(n_func):
    degree = np.random.randint(10)
    coefs, basis = random_polynom(degree)

    f_i = functools.partial(polynom, basis=basis)

    (x_train, y_train), (x_test, y_test) = generate_data_from_f(f_i)

    # loop on generated networks:
    for i in range(n_networks):
        units_i = networks[i]["units"]
        activations_i = networks[i]["activations"]
        # make i-th network
        model_i = generate_nn(units_i, activations_i)

        # fit/evaluate
        model_i = fit_nn(model_i, x_train, y_train)

        score = evaluate_model(model_i, x_test, y_test)
        print(f"f_{j} - score[{i}]: {score:.3f}")

        # save element in buffer

        element = dict(
            coefs=coefs.tolist(),
            x_train=x_train.tolist(),
            y_train=y_train.tolist(),
            x_test=y_train.tolist(),
            y_test=y_test.tolist(),
            score=score,
            network=dict(nunits=units_i, activations=activations_i),
        )

        data.append(element)

t2 = time.time()

print("Time: ", t2 - t1)
# %%

# save json

with open("data_polynome.json", "w") as f:
    json.dump(data, f)


# Goal of the story: generate/choose a good neural network for my current dataset (x,y)
# in one shot


# 10 functions <=> (x, y)
# 5 neural networks
# r possibles j'en ai 5 * 10 donc => (s, a, r)

# s \in R^{len(x)}

# Dataframe
# x | y | neural network | r2
# [-1,1] | f_0(x) | 0 | 0.9
# ... | f_1(x) | 0 | 0.8
# ... | f_0(x) |
# %%
