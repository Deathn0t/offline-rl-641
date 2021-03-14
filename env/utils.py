import numpy as np
import json
import copy


def sort_by_x(X, inplace=True):

    if inplace:
        X_ = X
    else:
        X_ = copy.deepcopy(X)
    
    for i, data in enumerate(X_):
        middle_point = int(len(data)/2)
        x = data[:middle_point]
        y = data[middle_point:]
        sorted_xy = np.asarray([[a, x] for a, x in sorted(
            zip(data[:middle_point], data[middle_point:]))])
        new_data = np.concatenate(
            (sorted_xy[:, 0], sorted_xy[:, 1]), axis=None)
        X_[i] = new_data
    return X_


def load_data(data_path, full=False):

    with open(data_path, "r") as f:
        data = json.load(f)
    # # keep only winning networks
    X, y, r = [], [], []
    n_networks = len(np.unique(data["actions"]))
    n_func = len(data["states"]) // n_networks
    # # print(n_networks, n_func)
    for f_i in range(n_func):
        scores_i = []
        actions_i = []
        for n_i in range(n_networks):
            i = f_i * n_networks + n_i
            scores_i.append(data["scores"][i])
            actions_i.append(data["actions"][i])
        X.append(data["states"][i])
        y.append(actions_i)
        r.append(scores_i)
    X = np.array(X)
    y = np.array(y)
    r = np.array(r)

    # print(X.shape)
    # print(y.shape)
    # print(r.shape)
    # fix action ordering
    idx_order = np.argsort(y[0])
    y = y[:, idx_order]
    r = r[:, idx_order]

    if not full:
        idxs = np.arange(X.shape[0])
        test_split = int(0.33 * len(idxs))
        np.random.shuffle(idxs)

        idxs_train = idxs[:-test_split]
        idxs_test = idxs[-test_split:]

        X_train = X[idxs_train]
        y_train = y[idxs_train]
        r_train = r[idxs_train]

        X_test = X[idxs_test]
        y_test = y[idxs_test]
        r_test = r[idxs_test]

        return sort_by_x(X_train), y_train, sort_by_x(X_test), y_test, r_train, r_test
    else:
        return X, y, r
