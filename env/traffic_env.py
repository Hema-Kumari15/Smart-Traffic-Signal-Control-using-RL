import numpy as np
import random

class TrafficEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0
        
        # cars in each lane
        self.north = random.randint(0, 20)
        self.south = random.randint(0, 20)
        self.east = random.randint(0, 20)
        self.west = random.randint(0, 20)

        self.signal = 0  # 0: NS green, 1: EW green
        return self.get_state()

    def get_state(self):
        return np.array([
            self.north, self.south,
            self.east, self.west,
            self.signal
        ], dtype=np.float32)

    def step(self, action):
        reward = 0

        # Action: 0 = NS green, 1 = EW green
        self.signal = action

        # cars passing
        if self.signal == 0:
            passed_ns = min(self.north, 5) + min(self.south, 5)
            self.north -= min(self.north, 5)
            self.south -= min(self.south, 5)
            reward += passed_ns

        else:
            passed_ew = min(self.east, 5) + min(self.west, 5)
            self.east -= min(self.east, 5)
            self.west -= min(self.west, 5)
            reward += passed_ew

        # new cars arrive
        self.north += random.randint(0, 5)
        self.south += random.randint(0, 5)
        self.east += random.randint(0, 5)
        self.west += random.randint(0, 5)

        # penalty for waiting
        wait_penalty = self.north + self.south + self.east + self.west
        reward -= wait_penalty * 0.1

        self.time += 1
        done = self.time >= 50

        return self.get_state(), reward, done, {}
