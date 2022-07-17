from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Dot, Softmax
from keras import activations
from keras.losses import Huber
from keras.optimizers import Adam
from dqn.modified_tensorboard import ModifiedTensorBoard

from collections import deque

import time
import random
import numpy as np
import tensorflow as tf
# from dqn.tensordot_layer import Tensordot
from keras.layers import Lambda

class DQNAgent:
    def __init__(self, obs_size, word_list, hidden_size,
                 replay_mem_size=1_000, min_replay_mem_size=1_000, minibatch_size=64, update_target_every=10, discount=0.99):
        self.word_width = 26*5
        self.obs_size = obs_size
        self.hidden_size = hidden_size

        word_array = np.zeros((self.word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1

        self.words = word_array

        self.replay_memory = deque(maxlen=replay_mem_size)
        self.min_replay_mem_size = min_replay_mem_size
        self.minibatch_size = minibatch_size
        self.update_target_every = update_target_every
        self.discount = discount
        
        # Main model
        self.model = self.create_model()
        
        # Target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.tensorboard = ModifiedTensorBoard()

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def tensordot(self, x):
        _x = tf.cast(x, dtype='float64')
        return tf.tensordot(_x, self.words, ((1,), (0,)))
        
    def create_model(self):
        input_layer = Input(shape=(self.obs_size, ))
        enc_layer = Dense(self.hidden_size, activation='linear')(input_layer)
        enc_layer = Activation(activations.relu)(enc_layer)
        enc_layer = Dense(self.hidden_size, activation='linear')(enc_layer)
        enc_layer = Activation(activations.relu)(enc_layer)
        enc_layer = Dense(self.word_width, activation='linear')(enc_layer)
        enc_layer = Lambda(self.tensordot)(enc_layer)
        enc_layer = Softmax()(enc_layer)

        model = Model(inputs=input_layer, outputs=enc_layer)

        print(model.summary())

        model.compile(loss=Huber(), optimizer=Adam(lr=1e-2), metrics=['accuracy'])
        
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        state_new = np.array(state).reshape(-1, *state.shape)

        # print(self.model.predict(state_new)[0])
        
        return self.model.predict(state_new)[0]

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_mem_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
