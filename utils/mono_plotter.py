import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import itertools

log_prefix = "./log/monopoly_Greedy_0.1_2e-07_0.95_1_1000_0_20230530_214418_run"
HORIZON = 1
M = 15

with open(log_prefix + ".pkl", "rb") as f:
    data = pkl.load(f)

# Plot Q-table.
fig = plt.figure(figsize=(50,50))
Q_table_0 = data["player_0"]
x = np.array(range(len(Q_table_0)))
for i in range(M):
    for j in range(M):
        ax = fig.add_subplot(M,M,i*M+j+1)
        for a in range(M):
            y_a = np.array([x[(i,j)][a] for x in Q_table_0])
            ax.plot(x, y_a, label=f"{a}")
fig.savefig(log_prefix + "_Q-table.png", dpi=200)