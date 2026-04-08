import numpy as np
from env.traffic_env import TrafficEnv

env = TrafficEnv()
scores = []

for _ in range(20):
    state = env.reset()
    total = 0

    for _ in range(100):
        action = (np.random.randint(0,2), np.random.randint(5,30))
        state, reward, done, _ = env.step(action)
        total += reward
        if done:
            break

    scores.append(total)

print("Baseline:", np.mean(scores))
