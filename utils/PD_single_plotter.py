import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import itertools

log_prefix = "./log/run_PD_LSTM_tit_0.1_32_16_2_64_20230503_135358"
HORIZON = 1
actions = np.array([0, 1])
states = list(itertools.product(actions, repeat=2))

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
x = np.array(range(len(data["player_0"])))
fig = plt.figure(figsize=(16,4))

i = 0
for s in states:
    i = i + 1

    ax = fig.add_subplot(1,len(states),i)
    y_0 = np.array([x[s][0] if s in x else 0 for x in data["player_0"]])
    y_1 = np.array([x[s][1] if s in x else 0 for x in data["player_0"]])
    ax.plot(x, y_0, label="C")
    ax.plot(x, y_1, label="D")
    ax.legend(loc="lower left")
    ax.set_xlabel(f"state = {s}")
fig.savefig(log_prefix + "_Q-table.png", dpi=200)