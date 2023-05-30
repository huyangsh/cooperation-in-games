import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import itertools

HORIZON  = 1
LOG_FREQ = 50000
actions = np.array([0, 1])
states = list(itertools.product(actions, repeat=2))

log_prefix = "./log/PD_LSTM/run_PD_Greedy_0.1_0.05_0.9_1_20230502_143518"
with open(log_prefix + ".pkl", "rb") as f:
    data = pkl.load(f)

# Plot stage rewards.
y_0 = np.array(data["simulation"]["reward_0"])
y_1 = np.array(data["simulation"]["reward_1"])
x = np.array(range(len(y_0)))
y_0_avg = y_0.cumsum() / (x+1)
y_1_avg = y_1.cumsum() / (x+1)

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(2,1,1)
ax.plot(x, y_0, label="Player 0")
ax.plot(x, y_1, label="Player 1")

ax = fig.add_subplot(2,1,2)
ax.plot(x, y_0_avg, label="Player 0")
ax.plot(x, y_1_avg, label="Player 1")
fig.savefig(log_prefix + "_rewards.png", dpi=200)


# Plot Q-table.
x = np.array(range(len(data["player_0"]))) * LOG_FREQ
fig = plt.figure(figsize=(16,4))

i = 0
for s in states:
    i = i + 1

    ax = fig.add_subplot(2,len(states),i)
    y_0 = np.array([x[s][0] for x in data["player_0"]])
    y_1 = np.array([x[s][1] for x in data["player_0"]])
    ax.plot(x, y_0, label="C")
    ax.plot(x, y_1, label="D")
    ax.legend(loc="lower left")
    if i == 1: ax.set_ylabel("Player 0")

    ax = fig.add_subplot(2,len(states),len(states)+i)
    y_0 = np.array([x[s][0] for x in data["player_1"]])
    y_1 = np.array([x[s][1] for x in data["player_1"]])
    ax.plot(x, y_0, label="C")
    ax.plot(x, y_1, label="D")
    ax.legend(loc="lower left")
    ax.set_xlabel(f"state = {s}")
    if i == 1: ax.set_ylabel("Player 1")
fig.savefig(log_prefix + "_Q-table.png", dpi=200)