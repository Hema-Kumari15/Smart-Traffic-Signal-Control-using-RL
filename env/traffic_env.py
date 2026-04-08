import numpy as np
import random

class TrafficEnv:
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.eval_mode = False
        self.reset()

    def reset(self):
        self.step_count = 0
        self.queues = np.random.randint(0, 20, 4)
        self.waiting = np.zeros(4)
        self.time = random.randint(0, 23)
        self.phase = 0
        self.emergency = random.choice([0, 1])
        return self._get_state()

    def _get_state(self):
        return np.concatenate([
            self.queues,
            self.waiting,
            [self.emergency, self.time, self.phase]
        ]).astype(np.float32)

    def step(self, action):
        phase, duration = action
        cars_passed = 0

        for _ in range(duration):
            if phase == 0:
                lanes = [0, 1]
            else:
                lanes = [2, 3]

            for i in lanes:
                passed = min(self.queues[i], 2)
                self.queues[i] -= passed
                cars_passed += passed

            if not self.eval_mode:
                self.queues += np.random.randint(0, 3, 4)

            self.waiting += self.queues

        total_wait = np.sum(self.waiting)
        queue_penalty = np.sum(self.queues)
        fairness = np.std(self.queues)

        reward = (
            -0.1 * total_wait
            -0.05 * queue_penalty
            +0.2 * cars_passed
            -0.1 * fairness
        )

        if self.emergency == 1 and phase == 0:
            reward += 5

        self.phase = phase
        self.time = (self.time + 1) % 24
        self.step_count += 1

        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, {}
