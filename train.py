import numpy as np
import matplotlib.pyplot as plt
from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent

env = TrafficEnv()
state_size = 11
action_size = 3

agent = DQNAgent(state_size, action_size)

episodes = 100
rewards_history = []

for e in range(episodes):
    state = env.reset()
    state = state.reshape(1, state_size)

    total_reward = 0

    for _ in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape(1, state_size)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    if len(agent.memory) > 32:
        agent.replay(32)

    rewards_history.append(total_reward)
    print(f"Episode {e+1}: {total_reward}")

plt.plot(rewards_history)
plt.savefig("results.png")
