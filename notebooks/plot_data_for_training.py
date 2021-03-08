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

# Distribution of scores for all networks
x = [max(el["score"], -1) for el in data]

plt.figure()

plt.hist(x, density=True)
plt.xlabel("$R^2$")

plt.show()
# %%

# Distribution of scores per network
def hash_network(config):
    return (
        "_".join(map(str, config["nunits"]))
        + "-"
        + "_".join(map(str, config["activations"]))
    )


score_per_network = {}
for el in data:
    key = hash_network(el["network"])
    val = max(el["score"], -1)

    score_per_network[key] = score_per_network.get(key, []) + [val]

plt.figure()

for i, (k, v) in enumerate(score_per_network.items()):
    plt.subplot(3, 2, i + 1)
    plt.hist(v)
    plt.xlabel(f"$R^2$ for {k}")

plt.tight_layout()
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


# {'9_5-relu_tanh': 325,
#  '4_4_1_10_10_9-relu_tanh_relu_tanh_tanh_relu': 288,
#  '8_10-tanh_relu': 255,
#  '9_3_1_3-tanh_tanh_relu_tanh': 24,
#  '7-tanh': 8}

# %%

# mapping networks to class
networks = list(winning_networks.keys())
networks_to_id = {v:i for i,v in enumerate(networks)}
id_to_networks = {i:v for i,v in enumerate(networks)}

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
