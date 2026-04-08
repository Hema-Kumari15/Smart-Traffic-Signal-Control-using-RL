class PPOAgent:
    def act(self, state):
        actions = []

        for i in range(len(state["queues"])):
            queues = state["queues"][i]

            ns = queues[0] + queues[1]
            ew = queues[2] + queues[3]

            phase = 0 if ns > ew else 1
            duration = min(30, max(5, int(sum(queues)/2)))

            actions.append((phase, duration))

        return actions
