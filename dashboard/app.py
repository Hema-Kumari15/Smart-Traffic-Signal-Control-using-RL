import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from env.traffic_env import TrafficEnv

env = TrafficEnv()

if "state" not in st.session_state:
    st.session_state.state = env.reset()
    st.session_state.rewards = []

state = st.session_state.state

st.title("🚦 Traffic RL Dashboard")

st.write({
    "North": int(state[0]),
    "South": int(state[1]),
    "East": int(state[2]),
    "West": int(state[3]),
    "Emergency": int(state[8])
})

phase = st.selectbox("Phase", ["NS", "EW"])
duration = st.slider("Duration", 5, 30)

if st.button("Step"):
    action = (0 if phase=="NS" else 1, duration)
    next_state, reward, done, _ = env.step(action)

    st.session_state.state = next_state
    st.session_state.rewards.append(reward)

    st.write("Reward:", reward)

if len(st.session_state.rewards) > 1:
    fig, ax = plt.subplots()
    ax.plot(st.session_state.rewards)
    st.pyplot(fig)
