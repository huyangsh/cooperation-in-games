import random, itertools
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from game import MatrixGame
from player import EpsGreedyPlayer, LSTMPlayer, LSTMBatchPlayer, TitForTatPlayer


# Configurations.
# ======================================
ALPHA   = 0.1
GAMMA   = 0.9
BETA    = 1e-4
EPS     = 0.05

HORIZON     = 32
HIDDEN_SIZE = 16
NUM_LAYERS  = 1
LR          = 1e-2
LR_DECAY    = 0.9995
BATCH_SIZE  = 1

T           = 1000000
LOG_FREQ    = 50000
CLEAR_FREQ  = 200000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
# **************************************


# Players and action space.
# ======================================
actions = np.array([0, 1])

PLAYER_TYPE = 0
if PLAYER_TYPE == 1:
    log_prefix = f"./log/run_PD_Greedy_{ALPHA}_{EPS}_{GAMMA}_{HORIZON}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = EpsGreedyPlayer(
        pid=0, actions=actions,
        alpha=ALPHA, eps=EPS, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ
    )
    player_1 = EpsGreedyPlayer(
        pid=1, actions=actions,
        alpha=ALPHA, eps=EPS, gamma=GAMMA, horizon=HORIZON,
        log_freq=LOG_FREQ
    )
elif PLAYER_TYPE == 2:
    log_prefix = f"./log/run_PD_LSTM_{ALPHA}_{HORIZON}_{HIDDEN_SIZE}_{NUM_LAYERS}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = LSTMPlayer(
        pid=0, actions=actions,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        lr=LR, lr_decay=LR_DECAY, device=device, log_freq=LOG_FREQ
    )
    player_1 = LSTMPlayer(
        pid=1, actions=actions,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        lr=LR, lr_decay=LR_DECAY, device=device, log_freq=LOG_FREQ
    )
elif PLAYER_TYPE == 3:
    log_prefix = f"./log/run_PD_LSTM_Batch_{ALPHA}_{HORIZON}_{HIDDEN_SIZE}_{NUM_LAYERS}_{BATCH_SIZE}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = LSTMBatchPlayer(
        pid=0, actions=actions,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        lr=LR, lr_decay=LR_DECAY, device=device, batch_size=BATCH_SIZE, log_freq=LOG_FREQ
    )
    player_1 = LSTMBatchPlayer(
        pid=1, actions=actions,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        lr=LR, lr_decay=LR_DECAY, device=device, batch_size=BATCH_SIZE, log_freq=LOG_FREQ
    )
elif PLAYER_TYPE == 4:
    PERIOD   = 2
    T        = 100000
    LOG_FREQ = 50

    log_prefix = f"./log/run_PD_LSTM_tit_{ALPHA}_{HORIZON}_{HIDDEN_SIZE}_{NUM_LAYERS}_{PERIOD}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = LSTMPlayer(
        pid=0, actions=actions,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, horizon=HORIZON,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        lr=LR, lr_decay=LR_DECAY, device=device, log_freq=LOG_FREQ
    )
    player_1 = TitForTatPlayer(
        pid=1, C_id=0, C_action=0, D_id=1, D_action=1, period=PERIOD
    )
else:
    assert False, "Invalid player type."
# **************************************


# Game simulator.
# ======================================
PD_game = MatrixGame(
    players = [player_0, player_1],
    reward_matrix = [[(3,3), (0,5)], [(5,0), (1,1)]]
)

PD_game.run(
    T = T,
    clear_freq = CLEAR_FREQ,
    log_freq = LOG_FREQ,
    log_url = log_prefix + ".log",
    save_url = log_prefix + ".pkl",
)
# **************************************


# Plotting.
# ======================================
# stage rewards
y_0 = np.array(PD_game.log["reward_0"])
y_1 = np.array(PD_game.log["reward_1"])
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

# Q-table.
x = np.array(range(len(player_0.log["Q_table"])))
fig = plt.figure(figsize=(16,4))

i = 0
states = list(itertools.product(actions, repeat=2))
for s in states:
    i = i + 1

    ax = fig.add_subplot(2,len(states),i)
    y_0 = np.array([x[s][0] if s in x else 0 for x in player_0.log["Q_table"]])
    y_1 = np.array([x[s][1] if s in x else 0 for x in player_0.log["Q_table"]])
    ax.plot(x, y_0, label="C")
    ax.plot(x, y_1, label="D")
    ax.legend(loc="lower left")

    ax = fig.add_subplot(2,len(states),len(states)+i)
    y_0 = np.array([x[s][0] if s in x else 0 for x in player_1.log["Q_table"]])
    y_1 = np.array([x[s][1] if s in x else 0 for x in player_1.log["Q_table"]])
    ax.plot(x, y_0, label="C")
    ax.plot(x, y_1, label="D")
    ax.legend(loc="lower left")
    ax.set_xlabel(f"state = {s}")
fig.savefig(log_prefix + "_Q-table.png", dpi=200)
# **************************************