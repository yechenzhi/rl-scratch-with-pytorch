import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm
import time

import pickle5 as pickle
from tqdm.notebook import tqdm

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")

# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"     # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability
decay_rate = 0.0005            # Exponential decay rate for exploration prob


def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_num = random.uniform(0,1)
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    else:
        action = env.action_space.sample()
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(decay_rate * episode)
        state, info = env.reset()
        terminated, truncated = False, False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * (np.max(Qtable[new_state][:])) - Qtable[state][action])

            if terminated or truncated:
                break
            state = new_state
    return Qtable


def evaluate_agent(n_eval_episodes, max_steps, Qtable, seed):
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed = seed[episode])
        else:
            state, info = env.reset()

        truncated, terminated = False, False
        total_rewards_ep = 0

        for step in range(max_steps):
            action = greedy_policy(Qtable, state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

if __name__ == '__main__':
    state_space = env.observation_space.n
    action_space = env.action_space.n
    Qtable_frozenlake = initialize_q_table(state_space, action_space)
    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

    mean_reward, std_reward = evaluate_agent(n_eval_episodes, max_steps, Qtable_frozenlake, eval_seed)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")




