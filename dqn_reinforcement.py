import time
import random
import numpy as np
import tensorflow as tf
import os

from dqn.agent import DQNAgent
from dqn.wordle_env import WordleEnv
from tqdm import tqdm
from load_words import _load_words

VALID_WORDS_PATH = 'words.txt'

DISCOUNT = 0.99

EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
MIN_REWARD = 10

MODEL_NAME = '2x256'

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

words = _load_words(10)

env = WordleEnv(words, 6)

agent = DQNAgent(env.observation_space_size, words, 256)

ep_rewards = []

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            q_table = np.asarray(agent.get_qs(current_state))
            argmax_layer = np.dot(q_table, agent.words)
            action = np.argmax(argmax_layer)
        else:
            # Get random action
            action = np.random.randint(0, env.action_space_size)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    

        
