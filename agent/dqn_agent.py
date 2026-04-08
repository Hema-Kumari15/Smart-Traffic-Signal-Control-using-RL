import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 0.001

        self.model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_size, activation='linear')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mse'
        )

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 1), random.randint(5, 30)

        q_vals = self.model.predict(state, verbose=0)[0]
        phase = np.argmax(q_vals[:2])
        duration = int(np.clip(q_vals[2] * 10, 5, 30))
        return phase, duration

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)

        for s, a, r, ns, d in minibatch:
            target = r
            if not d:
                target += self.gamma * np.max(self.model.predict(ns, verbose=0)[0])

            target_f = self.model.predict(s, verbose=0)
            target_f[0][0] = target  # simplified update

            self.model.fit(s, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
