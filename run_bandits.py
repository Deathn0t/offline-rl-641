from env.utils import load_data
import numpy as np
from tqdm import tqdm
from pprint import pprint
from matplotlib import pyplot as plt

import os
import json

from td3 import Normal_distribution, Bandit, UCB, IMED


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

def klGaussian(mean_1, mean_2, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return ((mean_1-mean_2)**2)/(2*sig2)

for i_fn in tqdm(range(n_functions)):
    rewards_i = r[i_fn, :]
    distributions_i = [Normal_distribution(r_arm, std=0.1) for r_arm in rewards_i]
    bandit_i = Bandit(distributions_i)

    # algo = UCB(bandit_i, inv_delta)
    algo = IMED(bandit_i, klGaussian)
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


fig, ax = plt.subplots(figsize=(10, 8))

for regret_i in regrets:
    ax.plot(regret_i, alpha=0.6)

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Cumulative regret', fontsize=12)

plt.title(algo.name, fontsize=14)

fig.tight_layout()
    
plt.savefig(os.path.join(RESULTS_PATH, f"{algo.name}_regrets.pdf"), dpi=1000)
plt.show()
