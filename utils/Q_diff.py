import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import itertools

def Q_diff(Q1, Q2):
    keys = Q1.keys()
    assert len(set(keys) - set(Q2.keys())) == 0

    diff = 0
    for key in keys:
        diff += np.linalg.norm(Q1[key] - Q2[key]) ** 2
    return np.sqrt(diff)

log_prefix = "./log/mono_paper_unbounded/run_mono_Greedy_0.1_2e-05_0.95_1_20230502_235350"
with open(log_prefix + ".pkl", "rb") as f:
    data = pkl.load(f)
Q_tables_0 = data["player_0"]
Q_tables_1 = data["player_1"]

Q_diff_0 = [0]
for i in range(len(Q_tables_0)-1):
    Q_diff_0.append(Q_diff(Q_tables_0[i+1], Q_tables_0[i]))

Q_diff_1 = [0]
for i in range(len(Q_tables_1)-1):
    Q_diff_1.append(Q_diff(Q_tables_1[i+1], Q_tables_1[i]))

print(Q_diff_0, Q_diff_1)
exit()

x = np.arange(len(Q_diff_0))
y_0 = np.array(Q_diff_0, dtype=np.float32)
y_1 = np.array(Q_diff_1, dtype=np.float32)
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)
ax.plot(x, y_0, label="Player 0")
ax.plot(x, y_1, label="Player 1")
ax.legend()
fig.savefig(log_prefix + "_Q-diff.png", dpi=200)
