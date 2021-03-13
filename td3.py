import numpy as np
from numpy import array, cumsum, where, amin, amax, sqrt, log, eye, sum
from numpy import max as maximum
from numpy import zeros, mean, std, quantile, arange
from numpy.random import normal, binomial, randint

########################################
#            Distributions             #
########################################


class Normal_distribution:
    def __init__(self, mean, std=1.):
        self.mean = mean
        self.std = std

    def __repr__(self):
        return f"Normal distribution (mean={self.mean:.3f}, std={self.std:.3f})"

    def sample(self):
        return normal(loc=self.mean, scale=self.std)


class Bernoulli_distribution:
    def __init__(self, mean):
        self.mean = mean

    def __repr__(self):
        return f"Bernoulli distribution (mean={self.mean:.3f})"

    def sample(self):
        return binomial(1, self.mean)


########################################
#               Bandit                 #
########################################


class Bandit:
    def __init__(self, distributions):
        self.distributions = distributions
        self.N_arms = len(distributions)
        self.rewards = array(
            [distributions[i].mean for i in range(self.N_arms)])
        self.regrets = maximum(self.rewards) - self.rewards

    def __str__(self):
        return f"Bandit({self.distributions})"

    def __repr__(self):
        return f"Bandit({self.distributions})"

    def sample(self, idx):
        return self.distributions[idx].sample()


########################################
#             Strategies               #
########################################


class UCB:
    def __init__(self, bandit, inv_delta, sigma=1):
        self.bandit = bandit
        self.inv_delta = inv_delta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.means = array([self.bandit.sample(i)
                            for i in range(self.bandit.N_arms)])
        self.nbr_pulls = array([1 for _ in range(self.bandit.N_arms)])
        self.time = self.bandit.N_arms
        self.regrets = [self.bandit.regrets[i]
                        for i in range(self.bandit.N_arms)]

    def __str__(self):
        res = f"UCB bandit algorithm - time step = {self.time} - cum. regret = {sum(self.regrets):.3f}\n"
        for i in range(self.bandit.N_arms):
            res += "  "
            res += str(self.bandit.distributions[i])
            res += " : "
            res += f"est. mean = {self.means[i]:.3f} - "
            res += f"nbr. pulls = {self.nbr_pulls[i]}\n"
        return res

    def play(self, idx):
        reward = self.bandit.sample(idx)
        self.time += 1
        self.means[idx] = (self.means[idx]*self.nbr_pulls[idx] +
                           reward)/(self.nbr_pulls[idx]+1)
        self.nbr_pulls[idx] += 1

    def compute_idx(self):
        # Exploration bonuses
        bonus = self.sigma * \
            sqrt(2 * log(self.inv_delta(self.time)) / self.nbr_pulls)
        # Indexes
        indexes = self.means + bonus
        return indexes

    def step(self):
        indexes = self.compute_idx()
        arms = where(indexes == amax(indexes))[0]
        under_played_arm = where(
            self.nbr_pulls[arms] == amin(self.nbr_pulls[arms]))[0]
        under_played_arm = under_played_arm[randint(len(under_played_arm))]
        under_played_arm = arms[under_played_arm]

        self.play(under_played_arm)
        self.regrets.append(self.bandit.regrets[under_played_arm])

    def fit(self, horizon, reset=True):
        if reset:
            self.reset()
        for _ in range(horizon):
            self.step()

    def cumulative_regret(self):
        return cumsum(self.regrets)


class IMED:
    def __init__(self, bandit, kullback):
        self.name = "IMED"
        self.bandit = bandit
        self.kl = kullback
        self.reset()

    def reset(self):
        self.all_selected = False
        self.means = np.zeros(self.bandit.N_arms)
        self.nbr_pulls = np.zeros(self.bandit.N_arms, int)
        self.max_mean = 0.
        self.indexes = np.zeros(self.bandit.N_arms)
        self.cum_rewards = np.zeros(self.bandit.N_arms)
        self.time = 0
        self.regrets = []

    def __str__(self):
        res = f"IMED algorithm - time step = {self.time} - cum. regret = {sum(self.regrets):.3f}\n"
        for i in range(self.bandit.N_arms):
            res += "  "
            res += str(self.bandit.distributions[i])
            res += " : "
            res += f"est. mean = {self.means[i]:.3f} - "
            res += f"nbr. pulls = {self.nbr_pulls[i]}\n"
        return res

    def play(self, arm):
        reward = self.bandit.sample(arm)
        self.cum_rewards[arm] += reward
        self.time += 1
        self.means[arm] = (self.means[arm]*self.nbr_pulls[arm] +
                           reward)/(self.nbr_pulls[arm]+1)
        self.nbr_pulls[arm] += 1
        self.max_mean = max(self.means)

    def compute_idx(self):
        if self.all_selected:
            self.indexes = self.nbr_pulls * \
                self.kl(self.means, self.max_mean) + log(self.nbr_pulls)
        else:
            self.all_selected = True
            for a in range(self.bandit.N_arms):
                if self.nbr_pulls[a] > 0:
                    self.indexes[a] = self.nbr_pulls[a] * \
                        self.kl(self.means[a], self.max_mean) + \
                        log(self.nbr_pulls[a])
                else:
                    self.indexes[a] = 0
                    self.all_selected = False

    def choose_an_arm(self):
        # Compute (weak) indexes
        self.compute_idx()
        # Break ties
        arms = where(self.indexes == amin(self.indexes))[0]
        under_played_arm = where(
            self.nbr_pulls[arms] == amin(self.nbr_pulls[arms]))[0]
        under_played_arm = under_played_arm[randint(len(under_played_arm))]
        under_played_arm = arms[under_played_arm]
        return under_played_arm

    def step(self):
        arm = self.choose_an_arm()
        # Pull suboptimal arm
        self.play(arm)
        self.regrets.append(self.bandit.regrets[arm])

    def fit(self, horizon, reset=True):
        if reset:
            self.reset()
        for _ in range(horizon):
            self.step()

    def cumulative_regret(self):
        return cumsum(self.regrets)


class KLUCBp:
    def __init__(self, bandit, kullback_solver):
        self.name = "KLUCB+"
        self.bandit = bandit
        self.kls = kullback_solver
        self.reset()

    def reset(self):
        self.all_selected = False
        self.means = np.zeros(self.bandit.N_arms)
        self.nbr_pulls = np.zeros(self.bandit.N_arms, int)
        self.max_mean = 0.
        self.indexes = np.zeros(self.bandit.N_arms)
        self.cum_rewards = np.zeros(self.bandit.N_arms)
        self.time = 0
        self.regrets = []

    def __str__(self):
        res = f"KLUCB+ algorithm - time step = {self.time} - cum. regret = {sum(self.regrets):.3f}\n"
        for i in range(self.bandit.N_arms):
            res += "  "
            res += str(self.bandit.distributions[i])
            res += " : "
            res += f"est. mean = {self.means[i]:.3f} - "
            res += f"nbr. pulls = {self.nbr_pulls[i]}\n"
        return res

    def play(self, arm):
        reward = self.bandit.sample(arm)
        self.cum_rewards[arm] += reward
        self.time += 1
        self.means[arm] = (self.means[arm]*self.nbr_pulls[arm] +
                           reward)/(self.nbr_pulls[arm]+1)
        self.nbr_pulls[arm] += 1

    def compute_idx(self):
        arms = where(self.means == amax(self.means))[0]
        under_played_arm = where(
            self.nbr_pulls[arms] == amin(self.nbr_pulls[arms]))[0]
        under_played_arm = under_played_arm[randint(len(under_played_arm))]
        under_played_arm = arms[under_played_arm]
        nbr_pulls_leader = self.nbr_pulls[under_played_arm]

        for a in range(self.bandit.N_arms):
            if self.nbr_pulls[a] == 0:
                self.indexes[a] = np.inf
            elif nbr_pulls_leader <= self.nbr_pulls[a]:
                self.indexes[a] = self.means[a]
            else:
                level = (log(nbr_pulls_leader) -
                         log(self.nbr_pulls[a]))/self.nbr_pulls[a]
                self.indexes[a] = self.kls(self.means[a], level)

    def choose_an_arm(self):
        # Compute (weak) indexes
        self.compute_idx()
        # Break ties
        arms = where(self.indexes == amax(self.indexes))[0]
        under_played_arm = where(
            self.nbr_pulls[arms] == amin(self.nbr_pulls[arms]))[0]
        under_played_arm = under_played_arm[randint(len(under_played_arm))]
        under_played_arm = arms[under_played_arm]
        return under_played_arm

    def step(self):
        arm = self.choose_an_arm()
        # Pull suboptimal arm
        self.play(arm)
        self.regrets.append(self.bandit.regrets[arm])

    def fit(self, horizon, reset=True):
        if reset:
            self.reset()
        for _ in range(horizon):
            self.step()

    def cumulative_regret(self):
        return cumsum(self.regrets)
