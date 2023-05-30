import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from game import MonopolyGame
from player import AdaptGreedyPlayer, AdaptGreedyBatchPlayer


# Configurations.
# ======================================
ALPHA   = 0.1
BETA    = 2e-6
GAMMA   = 0.95
HORIZON = 1

T           = 10000000
LOG_FREQ    = 500000
BATCH_SIZE  = 1000
# **************************************


# Players and action space.
# ======================================
M       = 15
XI      = 0.1
PN      = 1.61338
PM      = 1.73153
actions = np.linspace(PN - XI*(PM-PN), PM + XI*(PM-PN), num=M)
print(actions)
"""actions = np.hstack([
    np.linspace(0, PN - XI*(PM-PN), num=100),
    np.linspace(PN - XI*(PM-PN), PM + XI*(PM-PN), num=M)
])"""

PLAYER_TYPE = 0
if PLAYER_TYPE == 1:
    log_prefix = f"./log/run_mono_Greedy_{ALPHA}_{BETA}_{GAMMA}_{HORIZON}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = AdaptGreedyPlayer(
        pid=0, actions=actions,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ,
    )
    player_1 = AdaptGreedyPlayer(
        pid=1, actions=actions,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ,
    )
elif PLAYER_TYPE == 2:
    log_prefix = f"./log/run_mono_batch_{ALPHA}_{BETA}_{GAMMA}_{HORIZON}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = AdaptGreedyBatchPlayer(
        pid=0, actions=actions, batch_size=BATCH_SIZE, 
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ,
    )
    player_1 = AdaptGreedyBatchPlayer(
        pid=1, actions=actions, batch_size=BATCH_SIZE, 
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ,
    )
else:
    assert False, "Invalid player type."
# **************************************


# Game simulator.
# ======================================
monopoly_game = MonopolyGame(
    players = [player_0, player_1],
    a = [2, 2],
    a0 = 1,
    mu = 0.5,
    c = [1, 1]
)

# Default: Q-table initialized to Q^*.
print(" "*5 + "|" + "".join([f"a_{a}".rjust(8) for a in range(len(actions))]))
print("-"*6 + " -------"*len(actions))
print("  Q* |", end="")
for a in range(len(actions)):
    r_init = 0
    for b in range(len(actions)):
        r_init += monopoly_game.reward_func([a,b])[0]
    r_init = r_init / (1-GAMMA) / len(actions)
    print(f"{r_init:.3f}".rjust(8), end="")

    for s_0 in range(len(actions)):
        for s_1 in range(len(actions)):
            player_0.Q_table[(s_0,s_1)][a] = r_init
            player_1.Q_table[(s_0,s_1)][a] = r_init
print()
# Optional: Q-table initialized to random.
'''for s_0 in range(len(actions)):
    for s_1 in range(len(actions)):
        player_0.Q_table[(s_0,s_1)] = np.random.rand(len(actions)) * 10'''

monopoly_game.run(
    T = T,
    clear_freq = 10*BATCH_SIZE,
    log_freq = LOG_FREQ,
    log_url = log_prefix + ".log",
    save_url = log_prefix + ".pkl",
)
# **************************************


# Plotting.
# ======================================
# Q-table
fig = plt.figure(figsize=(50,50))
Q_table_0 = player_0.log["Q_table"]
x = np.array(range(len(Q_table_0)))
for i in range(M):
    for j in range(M):
        ax = fig.add_subplot(M,M,i*M+j+1)
        for a in range(M):
            y_a = np.array([x[(i,j)][a] for x in Q_table_0])
            ax.plot(x, y_a, label=f"{a}")
fig.savefig(log_prefix + "_Q-table.png", dpi=200)
# **************************************