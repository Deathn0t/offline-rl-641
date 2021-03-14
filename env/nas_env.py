import gym
from gym import error, spaces, utils
from gym.utils import seeding

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import numpy as np
import json
import copy
import random

from .utils import sort_by_x


N_DISCRETE_ACTIONS = 5 # Number of networks

class NAS_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 data_path, 
                 reward_penalty=0.0, 
                 naction_ending=5, 
                 step_sampling=False, 
                 offline_max_remove_data_perc = 0.3, 
                 offline_max_remove_network = 2):
        super(NAS_Env, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
        
        self.data_path = data_path
        self.offline_max_remove_data_perc = offline_max_remove_data_perc
        self.offline_max_remove_network = offline_max_remove_network 
        
        self.X_train, self.y_train, self.X_test, self.y_test, self.r_train, self.r_test = self.load_data()
        
        self.format_offline_train()
        self.format_offline_test()
        
        self.reward_range = (-1,1)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(2, 80), dtype=np.float16)
        
        self.reward_penalty = reward_penalty
        self.naction_ending = naction_ending
        self.step_sampling = step_sampling
        
        self.set_train_mode()
        
        self.current_dataset_id  = 0
        
        self.set_current_dataset()
        
        self.current_mask = []
        self.current_step = 1
        self.current_observation = self.get_observation()
        
        self.history_action = []
        self.history_rewards = []
        self.history_max_possible_rewards = []
         
        
    def set_train_mode(self):
        
        self.datasets = self.X_train
        self.ys = self.y_train
        self.rewards = self.r_train
        
    def set_test_mode(self):
        
        self.datasets = self.X_test
        self.ys = self.y_test
        self.rewards = self.r_test 
        
    def set_current_dataset(self):
        
        self.current_dataset = self.datasets[self.current_dataset_id].reshape((2,80))
        self.current_ys = self.ys[self.current_dataset_id]
        self.current_rewards = self.rewards[self.current_dataset_id] 
        self.current_possible_actions = [i for i in range(5) if np.isnan(self.current_rewards[i]) == False]

    def step(self, action):
        
        done = (self.current_step >= self.naction_ending and len(np.unique(
            self.history_action[-self.naction_ending:])) == 1) or (self.current_step == self.current_dataset.shape[1] - 1)
        
        full_reward = self.current_rewards[action]
        discounted_reward = full_reward * np.exp(-self.current_step * self.reward_penalty)
        
        max_possible_reward = np.nanmax(self.current_rewards)
        discounted_max_possible_reward = max_possible_reward * np.exp(-self.current_step * self.reward_penalty)
        
        self.history_action.append(action)
        self.history_rewards.append(discounted_reward)
        self.history_max_possible_rewards.append(discounted_max_possible_reward)
        
        self.current_step +=1
        self.current_observation = self.get_observation()
        
        return self.current_observation, discounted_reward, done, ""
        
         
    def get_observation(self):
        
        step_mask = np.zeros(self.current_dataset.shape[1])
        
        if self.step_sampling:
            available_idxs = list(set(range(len(step_mask))).difference(set(self.current_mask)))
            chosen_idx = np.random.choice(available_idxs)
            self.current_mask.append(chosen_idx)
        else:
            self.current_mask = list(range(self.current_step))
        
        step_mask[self.current_mask] = 1.0
        
        new_obs = copy.deepcopy(self.current_dataset)
        
        new_obs[0, :] = new_obs[0,:] * step_mask
        new_obs[1, :] = new_obs[1,:] * step_mask
        return new_obs
         
    # Execute one time step within the environment
    
    def reset(self, train = False):
        
        if train == True:
            self.set_train_mode()
        else:
            self.set_test_mode()
            
        self.current_dataset_id  = random.randint(0, self.datasets.shape[0]-1)
        self.set_current_dataset()
        
        self.current_mask = []
        self.current_step = 1
        self.current_observation = self.get_observation()
        
        self.history_action = []
        self.history_rewards = []
        self.history_max_possible_rewards = []
        
    # Reset the state of the environment to an initial state
    
    def render(self, mode='human', close=False):
        print("Error 404")
    # Render the environment to the screen
    
    def load_data(self):
        
        with open(self.data_path, "r") as f:
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
    

    def format_offline_train(self):
        
        for i, (x, r) in enumerate(zip(self.X_train, self.r_train)):
    
            number_max_to_be_removed = int(self.offline_max_remove_data_perc * 80) 
            number_to_be_removed = np.random.randint(number_max_to_be_removed+1, size=1)[0]
    
            if number_to_be_removed > 0:
    
                index_to_be_removed = np.array(random.sample(range(0, 80), number_to_be_removed))
                index_to_be_removed2 = index_to_be_removed + 80
    
                total_index = np.concatenate((index_to_be_removed, index_to_be_removed2))
                x[total_index] = 0.0
        
            number_network_to_be_removed = np.random.randint(self.offline_max_remove_network+1, size=1)[0]
    
            if number_network_to_be_removed  > 0:
    
                index_network_to_be_removed = np.array(random.sample(range(0, 5), number_network_to_be_removed))
                r[index_network_to_be_removed] = None
            
    def format_offline_test(self):
        
        for i, (x, r) in enumerate(zip(self.X_test, self.r_test)):
    
            number_max_to_be_removed = int(self.offline_max_remove_data_perc * 80) 
            number_to_be_removed = np.random.randint(number_max_to_be_removed +1, size=1)[0]
    
            if number_to_be_removed > 0:
    
                index_to_be_removed = np.array(random.sample(range(0, 80), number_to_be_removed))
                index_to_be_removed2 = index_to_be_removed + 80
    
                total_index = np.concatenate((index_to_be_removed, index_to_be_removed2))
                x[total_index] = 0.0
        
            number_network_to_be_removed = np.random.randint(self.offline_max_remove_network+1, size=1)[0]
    
            if number_network_to_be_removed  > 0:
    
                index_network_to_be_removed = np.array(random.sample(range(0, 5), number_network_to_be_removed))
                r[index_network_to_be_removed] = None
        
        