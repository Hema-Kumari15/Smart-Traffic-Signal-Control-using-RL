from fastapi import FastAPI
from pydantic import BaseModel
from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent
import numpy as np

app = FastAPI()

env = TrafficEnv(num_intersections=2)
agent = PPOAgent()

class Action(BaseModel):
    actions: list

@app.post("/reset")
def reset():
    return {"state": env.reset()}

@app.post("/step")
def step(action: Action):
    state, reward, done, info = env.step(action.actions)
    return {
        "state": state,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.post("/auto_step")
def auto_step():
    state = env._get_state()
    actions = agent.act(state)
    state, reward, done, _ = env.step(actions)

    return {
        "actions": actions,
        "state": state,
        "reward": float(reward),
        "done": bool(done)
    }

@app.post("/metrics")
def metrics():
    total_wait = float(np.sum(env.waiting))
    throughput = float(np.sum(env.queues))
    fairness = float(np.std(env.queues))

    score = max(0, 1 - total_wait/1000)

    return {
        "total_wait": total_wait,
        "throughput": throughput,
        "fairness": fairness,
        "score": score
    }

@app.post("/set_eval")
def set_eval(mode: bool):
    env.set_eval_mode(mode)
    return {"eval_mode": mode}
