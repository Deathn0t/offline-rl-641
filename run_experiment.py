from experiment_params import configurations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from distutils.util import strtobool

import numpy as np

import json
import time
import random
import os
import copy
import argparse
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

from tqdm import tqdm

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space

import matplotlib.pyplot as plt

from env.nas_env import NAS_Env
from env.dqn import Transition, ReplayMemory, DQN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    HERE, *["notebooks", "polynome_ready_for_training.json"])
RESULTS_PATH = os.path.join(HERE, *["experiment_results"])
FIGURES_PATH = os.path.join(RESULTS_PATH, *["figures"])

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_topk(predicted_network, score_networks, k):

    in_topk = 0

    for predicted, scores in zip(predicted_network, score_networks):

        top_idx = np.argsort(scores)[-k:]

        if predicted in top_idx:
            in_topk += 1

    return in_topk/len(predicted_network)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_env(param_set):

    reward_penalty = param_set['reward_penalty']
    naction_ending = param_set['naction_ending']
    step_sampling = param_set['step_sampling']
    offline_max_remove_data_perc = param_set['offline_max_remove_data_perc']
    offline_max_remove_network = param_set['offline_max_remove_network']

    return NAS_Env(DATA_PATH,
                   reward_penalty=reward_penalty,
                   naction_ending=naction_ending,
                   step_sampling=step_sampling,
                   offline_max_remove_data_perc=offline_max_remove_data_perc,
                   offline_max_remove_network=offline_max_remove_network)


def plot_true_best(i_exp, env):
    plt.bar(x=list(range(5)), height=np.bincount(
        np.nanargmax(env.r_train, axis=1)))
    plt.title("TRAIN True best network distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_PATH, f"exp{i_exp}-train_true_distribution.pdf"), dpi=1000)
    plt.clf()

    plt.bar(x=list(range(5)), height=np.bincount(
        np.nanargmax(env.r_test, axis=1)))
    plt.title("TEST True best network distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_PATH, f"exp{i_exp}-test_true_distribution.pdf"), dpi=1000)
    plt.clf()


def plot_fit(i_exp, SHUFFLE_SAMPLES, num_episodes, episode_predicted_networks, episode_max_possible_rewards, episode_rewards, episode_durations, is_eval=False):

    plt.bar(x=list(range(5)), height=np.bincount(np.array(
        episode_predicted_networks), minlength=5)/len(episode_predicted_networks))
    plt.title("Best network distribution (fitted agent)")
    plt.xlabel("Architecture index")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_PATH, f"exp{i_exp}-arch_distribution-predict.pdf"), dpi=1000)
    plt.clf()

    filter_size = 11
    pad = int((filter_size - 1)/2)

    eval_flag = "EVAL" if is_eval else "FIT"

    x_steps = np.linspace(0, num_episodes - 1, num=num_episodes, dtype=int)
    if is_eval:
        x_steps = np.linspace(0, len(episode_rewards) - 1,
                              num=len(episode_rewards), dtype=int)


    plt.plot(x_steps, np.array(episode_max_possible_rewards),
             label="Episode Max Possible Cum rewards")
    plt.plot(x_steps, np.array(episode_rewards), label="Episode Cum rewards")
    envtype = "random" if SHUFFLE_SAMPLES else "sequential"
    plt.title(f"DQN ({envtype} samples)")
    plt.xlabel("Episode number")
    plt.ylabel("Episode rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_PATH, f"exp{i_exp}-{eval_flag}-maxreward-DQN-{envtype}.pdf"), dpi=1000)
    plt.clf()

    regret = np.array(episode_max_possible_rewards) - np.array(episode_rewards)

    plt.plot(x_steps, regret, label="Episode regret")
    plt.plot(x_steps[pad:-pad], moving_average(regret,
                                               filter_size), label="Episode regret (smoothed)")
    envtype = "random" if SHUFFLE_SAMPLES else "sequential"
    plt.title(f"DQN ({envtype} samples)")
    plt.xlabel("Episode number")
    plt.ylabel("Mean regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_PATH, f"exp{i_exp}-{eval_flag}-regret-DQN-{envtype}.pdf"), dpi=1000)
    plt.clf()

    plt.plot(np.cumsum(regret), label="Cumulative regret")
    envtype = "random" if SHUFFLE_SAMPLES else "sequential"
    plt.title(f"DQN ({envtype} samples)")
    plt.xlabel("Episode number")
    plt.ylabel("Cumulative regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_PATH, f"exp{i_exp}-{eval_flag}-cumregret-DQN-{envtype}.pdf"), dpi=1000)
    plt.clf()

    plt.plot(x_steps, episode_durations, c='orange',
             label="Episode duration in steps")
    plt.plot(x_steps[pad:-pad], moving_average(episode_durations,
                                               filter_size), label="Episode Duration (smoothed)")
    envtype = "random" if SHUFFLE_SAMPLES else "sequential"
    plt.title(f"DQN ({envtype} samples)")
    plt.xlabel("Episode number")
    plt.ylabel("Episode length")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(
        FIGURES_PATH, f"exp{i_exp}-duration-DQN-{envtype}.pdf"), dpi=1000)
    plt.clf()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    non_final_next_states = torch.reshape(
        non_final_next_states, (int(non_final_next_states.shape[0]/2), 160))

    state_batch = torch.reshape(torch.cat(batch.state), (BATCH_SIZE, 160))
    action_batch = torch.reshape(torch.stack(batch.action), (BATCH_SIZE, 1))
    reward_batch = torch.cat(batch.reward)
    #possible_actions = torch.stack(batch.possible_actions)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    #target_actions = target_net(non_final_next_states)

    # print(target_actions)
    #best_target_allowed_action  = []
    # for actions_values, possible_actions in zip(target_actions, batch.possible_actions):
    # print(actions_values[possible_actions])
    # print(torch.max(actions_values[possible_actions]))
    # print(best_target_allowed_action)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.float(
    ), expected_state_action_values.unsqueeze(1).float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, required=True)
    args = parser.parse_args()

    set_seed(args.seed)

    num_episodes = 1000

    trace_experiment = dict()
    paramgrid = ParameterGrid(configurations)
    for i_param, param_set in tqdm(enumerate(paramgrid)):
        trace_experiment[i_param] = dict()
        trace_experiment[i_param]['seed'] = args.seed
        trace_experiment[i_param]['params'] = param_set

        env = get_env(param_set)

        plot_true_best(i_param, env)

        policy_net = DQN(env)
        target_net = DQN(env)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters())
        memory = ReplayMemory(100000)

        total_steps_done = 0

        BATCH_SIZE = param_set['BATCH_SIZE']
        GAMMA = param_set['GAMMA']
        EPS_START = param_set['EPS_START']
        EPS_END = param_set['EPS_END']
        EPS_DECAY = param_set['EPS_DECAY']
        TARGET_UPDATE = param_set['TARGET_UPDATE']

        episode_predicted_networks = []
        episode_best_true_networks = []
        episode_true_networks_r2 = []
        episode_rewards = []
        episode_max_possible_rewards = []
        episode_durations = []

        def get_action(state):

            global total_steps_done
            sample = random.random()

            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * total_steps_done / EPS_DECAY)

            total_steps_done += 1

            if sample > eps_threshold:
                with torch.no_grad():
                    policy_net.eval()
                    actions = policy_net(state)
                    policy_net.train()

                    ranking = torch.argsort(actions, descending=True)[0]
                    for action_candidate in ranking:
                        if action_candidate in env.current_possible_actions:
                            return torch.tensor(action_candidate)
            else:
                return torch.tensor(np.random.choice(env.current_possible_actions))

        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and state
            env.reset(train=True)

            state = torch.from_numpy(env.current_observation)

            for i in range(80):

                action = get_action(state.flatten().unsqueeze(0))
                _, reward, done, _ = env.step(action)
                reward = torch.tensor([reward])

                next_state = torch.from_numpy(env.current_observation)
                # Observe new state

                if done:
                    next_state = None

                # Store the transition in memory

                success_push = memory.push(state,
                                           action,
                                           next_state,
                                           reward,
                                           torch.tensor(env.current_possible_actions))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                optimize_model()

                if done:

                    episode_predicted_network = env.history_action[-2].numpy()
                    episode_best_true_network = np.nanargmax(
                        env.current_rewards)
                    episode_reward = sum(env.history_rewards)
                    episode_max_possible_reward = sum(
                        env.history_max_possible_rewards)

                    episode_predicted_networks.append(
                        episode_predicted_network)
                    episode_true_networks_r2.append(env.current_rewards)
                    episode_best_true_networks.append(
                        episode_best_true_network)
                    episode_rewards.append(episode_reward)
                    episode_max_possible_rewards.append(
                        episode_max_possible_reward)
                    episode_durations.append(i + 1)

                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        top1 = calculate_topk(
            episode_predicted_networks[-200:], episode_true_networks_r2[-200:], k=1)
        top2 = calculate_topk(
            episode_predicted_networks[-200:], episode_true_networks_r2[-200:], k=2)

        trace_experiment[i_param]['top1'] = top1
        trace_experiment[i_param]['top2'] = top2

        plot_fit(i_param, param_set['step_sampling'], num_episodes, episode_predicted_networks,
                 episode_max_possible_rewards, episode_rewards, episode_durations)

        eval_episode_predicted_networks = []
        eval_episode_best_true_networks = []
        eval_episode_true_networks_r2 = []
        eval_episode_rewards = []
        eval_episode_max_possible_rewards = []
        eval_episode_durations = []

        def eval(eval_episode_predicted_networks, eval_episode_best_true_networks, eval_episode_true_networks_r2, eval_episode_rewards, eval_episode_max_possible_rewards, eval_episode_durations):
            policy_net.eval()
            # print(env.datasets.shape[0])
            for i_episode in tqdm(range(env.r_test.shape[0])):
                # Initialize the environment and state
                env.reset(train=False)
                state = torch.from_numpy(env.current_observation)
                for i in range(80):
                    actions = policy_net(state.flatten().unsqueeze(0))
                    action = torch.argmax(actions)
                    _, reward, done, _ = env.step(action)
                    reward = torch.tensor([reward])
                    next_state = torch.from_numpy(env.current_observation)
                    # Observe new state
                    if done:
                        next_state = None
                    # Move to the next state
                    state = next_state
                    # Perform one step of the optimization (on the target network)
                    if done:
                        eval_episode_predicted_network = env.history_action[-2].numpy(
                        )
                        eval_episode_best_true_network = np.nanargmax(
                            env.current_rewards)
                        eval_episode_reward = sum(env.history_rewards)
                        eval_episode_max_possible_reward = sum(
                            env.history_max_possible_rewards)
                        eval_episode_predicted_networks.append(
                            eval_episode_predicted_network)
                        eval_episode_true_networks_r2.append(
                            env.current_rewards)
                        eval_episode_best_true_networks.append(
                            eval_episode_best_true_network)
                        eval_episode_rewards.append(eval_episode_reward)
                        eval_episode_max_possible_rewards.append(
                            eval_episode_max_possible_reward)
                        eval_episode_durations.append(i + 1)
                        break

        eval(eval_episode_predicted_networks, eval_episode_best_true_networks, eval_episode_true_networks_r2, eval_episode_rewards, eval_episode_max_possible_rewards, eval_episode_durations)

        eval_episode_predicted_networks = np.nan_to_num(eval_episode_predicted_networks, copy=False)
        eval_episode_best_true_networks = np.nan_to_num(eval_episode_best_true_networks, copy=False)
        eval_episode_true_networks_r2 = np.nan_to_num(eval_episode_true_networks_r2, copy=False)
        eval_episode_rewards = np.nan_to_num(eval_episode_rewards, copy=False)
        eval_episode_max_possible_rewards = np.nan_to_num(eval_episode_max_possible_rewards, copy=False)
        eval_episode_durations = np.nan_to_num(eval_episode_durations, copy=False)


        plot_fit(i_param, param_set['step_sampling'], num_episodes, eval_episode_predicted_networks,
                 eval_episode_max_possible_rewards, eval_episode_rewards, eval_episode_durations, is_eval=True)
        
    
    with open(os.path.join(RESULTS_PATH, *["trace.json"]), "w") as f:
        json.dump(trace_experiment, f)
