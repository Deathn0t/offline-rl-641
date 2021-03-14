# %%

import os
import json
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))

# %%

data_path = os.path.join(HERE, "data_polynome_all.json")
with open(data_path, "r") as f:
    data = json.load(f)

# %%

def hash_network(config):
    return (
        "_".join(map(str, config["nunits"]))
        + "-"
        + "_".join(map(str, config["activations"]))
    )

# mapping networks to class
networks = [hash_network(data[i]["network"]) for i in range(5)]
networks_to_id = {v:i for i,v in enumerate(networks)}
id_to_networks = {i:v for i,v in enumerate(networks)}

# %%

# distribution of generated polynomes

x = [len(el["coefs"])-1 for el in data[::5]]

plt.figure()

plt.hist(x,bins=10,density=True)

plt.xlabel("Degrees")

plt.show()
# %%

# Distribution of scores for all networks
x = [max(el["score"], -1) for el in data]

plt.figure()

plt.hist(x, density=True)
plt.xlabel("$R^2$")

plt.show()
# %%

# Distribution of scores per network



score_per_network = {}
for el in data:
    key = hash_network(el["network"])
    val = max(el["score"], -1)

    score_per_network[key] = score_per_network.get(key, []) + [val]

plt.figure()

for i, (k, v) in enumerate(score_per_network.items()):
    plt.subplot(2, 3, networks_to_id[k] + 1)
    plt.hist(v, density=True)
    plt.xlabel(f"$R^2$ for {networks_to_id[k]}")
    plt.ylim(0,3.5)

plt.tight_layout()
plt.savefig("figures/score-distribution-per-network.pdf")
plt.show()


# %%

# check how many times each network wins

n_networks = 5
n_func = len(data) // n_networks

winning_networks = {}

for f_i in range(n_func):

    best_network = None
    best_score = -2
    for n_i in range(n_networks):

        i = f_i * n_networks + n_i

        score = data[i]["score"]
        if score > best_score:
            best_score = score
            best_network = hash_network(data[i]["network"])

    winning_networks[best_network] = winning_networks.get(best_network, 0) + 1


height = [winning_networks[n] for n in networks]

plt.figure()
plt.bar(list(map(lambda x: networks_to_id[x], networks)), height)
plt.xlabel("Neural Network ID")
plt.ylabel("Number of Wins")
plt.tight_layout()
plt.savefig("figures/number-of-wins-per-network.pdf")
plt.show()

# %%

# plot degree vs winning network

n_networks = 5
n_func = len(data) // n_networks

degrees, best_networks = [], []

for f_i in range(n_func):

    coefs = data[f_i*n_networks]["coefs"]
    degree = len(coefs)-1 + (np.norm(coefs))
    best_network = None
    best_score = -2
    for n_i in range(n_networks):

        i = f_i * n_networks + n_i

        score = data[i]["score"]
        if score > best_score:
            best_score = score
            best_network = hash_network(data[i]["network"])

    degrees.append(degree)
    best_networks.append(networks_to_id[best_network])

plt.figure()
plt.scatter(degrees, best_networks)
# plt.hist2d(degrees, best_networks)
plt.show()

# %%

# Create the dataset for the RL agent
# score, network, data
states, actions, scores = [], [], []
for el in data:
    state = el["x_train"] + el["y_train"]
    action = networks_to_id[hash_network(el["network"])]
    score = max(-1, el["score"]) # min value is -1

    states.append(state)
    actions.append(action)
    scores.append(score)

assert len(states) == len(actions) and len(actions) == len(scores)


# %%

# save dataset
dataset = dict(
    states=states,
    actions=actions,
    scores=scores
)

path_dataset = os.path.join(HERE, "polynome_ready_for_training.json")
with open(path_dataset, "w") as f:
    json.dump(dataset, f)



# %%
