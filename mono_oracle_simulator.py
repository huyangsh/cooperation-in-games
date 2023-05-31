import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from tqdm import tqdm

from game import MonopolyGame
from game.utils import logging
from player import AdaptGreedyPlayer, AdaptGreedyBatchPlayer


np.random.seed(0)
random.seed(0)

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
action_set = np.linspace(PN - XI*(PM-PN), PM + XI*(PM-PN), num=M)
print(action_set)
"""actions = np.hstack([
    np.linspace(1, PN - XI*(PM-PN), num=100),
    np.linspace(PN - XI*(PM-PN), PM + XI*(PM-PN), num=M)
])"""

PLAYER_TYPE = 1
if PLAYER_TYPE == 1:
    log_prefix = f"./log/run_mono_Greedy_{ALPHA}_{BETA}_{GAMMA}_{HORIZON}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = AdaptGreedyPlayer(
        pid=0, actions=action_set,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ,
    )
    player_1 = AdaptGreedyPlayer(
        pid=1, actions=action_set,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ,
    )
elif PLAYER_TYPE == 2:
    log_prefix = f"./log/run_mono_batch_{ALPHA}_{BETA}_{GAMMA}_{HORIZON}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = AdaptGreedyBatchPlayer(
        pid=0, actions=action_set, batch_size=BATCH_SIZE, 
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ,
    )
    player_1 = AdaptGreedyBatchPlayer(
        pid=1, actions=action_set, batch_size=BATCH_SIZE, 
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
print(" "*5 + "|" + "".join([f"a_{a}".rjust(8) for a in range(len(action_set))]))
print("-"*6 + " -------"*len(action_set))
print("  Q* |", end="")
for a in range(len(action_set)):
    r_init = 0
    for b in range(len(action_set)):
        r_init += monopoly_game.reward_func([a,b])[0]
    r_init = r_init / (1-GAMMA) / len(action_set)
    print(f"{r_init:.3f}".rjust(8), end="")

    for s_0 in range(len(action_set)):
        for s_1 in range(len(action_set)):
            player_0.Q_table[(s_0,s_1)][a] = r_init
            player_1.Q_table[(s_0,s_1)][a] = r_init
print()
# Optional: Q-table initialized to random.
'''for s_0 in range(len(actions)):
    for s_1 in range(len(actions)):
        player_0.Q_table[(s_0,s_1)] = np.random.rand(len(actions)) * 10'''


# Temporary: need to merge into game.
T = T
clear_freq = 10*BATCH_SIZE
eval_freq = 1e6
log_freq = LOG_FREQ
log_url = log_prefix + ".log"
save_url = log_prefix + ".pkl"
assert clear_freq >= 0, "Clearing frequency should be a non-negative integer."

history, history_st = [], 0
monopoly_game._init_log(log_url)

bar = tqdm(range(T))
for t in bar:
    # Reinitialize game.
    state = tuple(random.choices(action_set, k=2))

    actions = []
    for player in monopoly_game.players:
        player.update(t, state, monopoly_game.get_history)
        actions.append(player.play(t, state, monopoly_game.get_history))

    rewards = monopoly_game.reward_func(actions)
    history.append((state, actions, rewards, t))

    if log_freq > 0 and t % log_freq == 0: 
        with open(log_url, "a") as f:
            logging(f, player_0.get_print_data())
            logging(f, player_1.get_print_data())
    
    if (eval_freq > 0) and (t % eval_freq == 0):
        with open(log_url, "a") as f:
            logging(f, f"Eval at time step {t}.\n")
        monopoly_game.run(
            T = 1000,
            clear_freq = 0,
            log_freq = LOG_FREQ,
            log_url = log_prefix + ".log"
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