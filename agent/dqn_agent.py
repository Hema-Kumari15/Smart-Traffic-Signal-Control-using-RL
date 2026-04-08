import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 0.001

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        return model

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_vals = self.model.predict(state, verbose=0)
        return np.argmax(q_vals[0])

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)

        for s, a, r, ns, d in minibatch:
            target = r
            if not d:
                target += self.gamma * np.max(self.model.predict(ns, verbose=0)[0])

            target_f = self.model.predict(s, verbose=0)
            target_f[0][a] = target

            self.model.fit(s, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
