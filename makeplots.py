from env.utils import load_data, sort_by_x
import numpy as np
from tqdm import tqdm
from pprint import pprint
from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist
import copy

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, *["notebooks", "polynome_ready_for_training.json"])
RESULTS_PATH = os.path.join(HERE, *["notebooks", "figures"])

os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(42)

X, y, r = load_data(DATA_PATH, full=True)
sort_by_x(X, inplace=True)
# print (X.shape)

def plot_sampling(X, steps=3):

    n_samples = X.shape[1] // 2
    idx_toplot = np.random.randint(X.shape[0])

    # sort_by_x(X, inplace=True)

    xs_toplot = X[idx_toplot, :n_samples]
    ys_toplot = X[idx_toplot, n_samples:]

    idxs_shuffle=np.arange(n_samples)
    np.random.shuffle(idxs_shuffle)

    x_toplot = xs_toplot[idxs_shuffle]
    y_toplot = ys_toplot[idxs_shuffle]


    xrange = [np.amin(x_toplot), np.amax(x_toplot)*1.1]
    yrange = [np.amin(y_toplot), np.amax(y_toplot)*1.1]

    bins = np.linspace(1, n_samples, num=steps+1, dtype=int)

    fig, ax = plt.subplots(nrows=steps, ncols=2, figsize=(8, steps*2))

    for i in range(steps):
        pos_i = bins[i+1]
        for j in range(2):
            ax[i, j].set_ylim(yrange)
            ax[i, j].set_xlim(xrange)
            if j==0:
                x_ij = xs_toplot[:pos_i]
                y_ij = ys_toplot[:pos_i]
                ax[i, j].scatter(x_ij, y_ij)
                ax[i, j].set_title(f'Step {pos_i} (sequential)')
            else:
                x_ij = x_toplot[:pos_i]
                y_ij = y_toplot[:pos_i]
                ax[i, j].scatter(x_ij, y_ij)
                ax[i, j].set_title(f'Step {pos_i} (sampling)')
            
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f"episode_types.pdf"), dpi=1000)
    plt.show()
    plt.clf()

def plot_fndistance(X):

    n_samples = X.shape[1] // 2

    x = X[:, :n_samples]
    y = X[:, n_samples:]

    D = cdist(y, y, metric="minkowski")
    plt.hist(D.ravel(), bins=20, density=True)
    plt.title("Histogram of pair-wise distance between objective functions")
    plt.xlabel("Minkowski distance")
    plt.ylabel("Density")
    
    plt.savefig(os.path.join(RESULTS_PATH, f"polynomes-distances.pdf"), dpi=1000)
    plt.show()
    plt.clf()

def plot_reward_closeness(r):

    sorted_r = np.sort(r, axis=1)
    n_networks = sorted_r.shape[1]

    fig, ax = plt.subplots(nrows=1, ncols=n_networks-1, figsize=(3*n_networks, 4), sharey=True)

    ax[0].set_ylabel('Count', fontsize=14)
    ax[0].set_ylim([0, 500])

    for i in range(2, n_networks+1):
        range_topk = np.ptp(sorted_r[:,:i], axis=1).ravel()
        ax[i-2].set_xlim([0, 2.0])
        
        if i==n_networks:
            ax[i-2].set_title('All actions', fontsize=14)
        else:
            ax[i-2].set_title(f'Top-{i} actions', fontsize=14)
        ax[i-2].set_xlabel("Absolute reward difference")
        ax[i-2].hist(range_topk, bins=20, density=False)

    fig.suptitle("Distribution of absolute reward difference for top-k actions (all objective functions)", fontsize=16)
    
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_PATH, f"reward-hardness.pdf"), dpi=1000)
    plt.show()
    plt.clf()


    

if __name__=="__main__":
    # plot_sampling(X, steps=5)
    # plot_fndistance(X)
    # plot_reward_closeness(r)