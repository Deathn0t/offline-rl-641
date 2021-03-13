from env.utils import load_data
import numpy as np
from tqdm import tqdm
from pprint import pprint

import os
import json

from td3 import Normal_distribution, Bandit, UCB


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, *["notebooks", "polynome_ready_for_training.json"])
RESULTS_PATH = os.path.join(HERE, *["bandit_results"])

os.makedirs(RESULTS_PATH, exist_ok=True)

X, y, r = load_data(DATA_PATH, full=True)

# Train 1 bandit per function in X_train:

n_functions, n_arms = r.shape

print(X.shape)

regrets = []
means = []

def inv_delta(t):
    return (t+1)**3

for i_fn in tqdm(range(n_functions)):
    rewards_i = r[i_fn, :]
    distributions_i = [Normal_distribution(r_arm, std=0.1) for r_arm in rewards_i]
    bandit_i = Bandit(distributions_i)

    algo = UCB(bandit_i, inv_delta)
    algo.fit(X.shape[1]//2, reset=True)

    means.append(algo.means)
    regrets.append(algo.cumulative_regret())


# pprint(means)
# pprint(regrets)


# Save results
result_database = dict()

for i in range(X.shape[0]):
    result_database[i] = dict()
    result_database[i]["regret"] = list(regrets[i].astype(float))
    result_database[i]["bandit_means"] = list(means[i].astype(float))
    result_database[i]["true_means"] = list(r[i].astype(float))


with open(os.path.join(RESULTS_PATH, *[f"{algo.name}.json"]), "w") as f:
    json.dump(result_database, f)

print("Saved results to", RESULTS_PATH)


