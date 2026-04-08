import numpy as np
import random

class TrafficEnv:
    def __init__(self, max_steps=100, num_intersections=1):
        self.max_steps = max_steps
        self.num_intersections = num_intersections
        self.eval_mode = False
        self.reset()

    def reset(self):
        self.step_count = 0

        self.queues = np.random.randint(0, 20, (self.num_intersections, 4))
        self.waiting = np.zeros((self.num_intersections, 4))

        self.time = random.randint(0, 23)
        self.phase = [0] * self.num_intersections
        self.emergency = [random.choice([0, 1]) for _ in range(self.num_intersections)]

        return self._get_state()

    def _get_state(self):
        return {
            "queues": self.queues.tolist(),
            "waiting": self.waiting.tolist(),
            "emergency": self.emergency,
            "time": int(self.time),
            "phase": self.phase
        }

    def set_eval_mode(self, mode=True):
        self.eval_mode = mode

    def step(self, actions):
        total_passed = 0

        for idx, (phase, duration) in enumerate(actions):
            for _ in range(duration):
                lanes = [0,1] if phase == 0 else [2,3]

                for i in lanes:
                    passed = min(self.queues[idx][i], 2)
                    self.queues[idx][i] -= passed
                    total_passed += passed

                if not self.eval_mode:
                    self.queues[idx] += np.random.randint(0, 3, 4)

                self.waiting[idx] += self.queues[idx]

            self.phase[idx] = phase

        total_wait = float(np.sum(self.waiting))
        queue_penalty = float(np.sum(self.queues))
        fairness = float(np.std(self.queues))

        reward = (
            -0.1 * total_wait
            -0.05 * queue_penalty
            +0.2 * total_passed
            -0.1 * fairness
        )

        for i, e in enumerate(self.emergency):
            if e == 1 and self.phase[i] == 0:
                reward += 5

        self.time = (self.time + 1) % 24
        self.step_count += 1

        done = self.step_count >= self.max_steps

        return self._get_state(), float(reward), bool(done), {}
