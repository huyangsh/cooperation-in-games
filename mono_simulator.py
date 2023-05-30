import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

from game import MonopolyGame, MonopolyTrajectoryRunner
from player import AERPlayer, AdaptGreedyPlayer, AdaptGreedyBatchPlayer


np.random.seed(0)
random.seed(0)

# Configurations.
# ======================================
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])

parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--beta", default=2e-5, type=float)
parser.add_argument("--gamma", default=0.95, type=float)
parser.add_argument("--horizon", default=1, type=int)

parser.add_argument("--player_type", type=int)
parser.add_argument("--batch_size", default=1000, type=int)

parser.add_argument("--T", default=int(1e7), type=int)
parser.add_argument("--log_freq", default=int(5e5), type=int)

args = parser.parse_args()
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

if args.player_type == 0:
    log_prefix = f"./log/run_mono_AER_{args.alpha}_{args.beta}_{args.gamma}_{args.horizon}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = AERPlayer(
        pid=0, actions=actions,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
    player_1 = AERPlayer(
        pid=1, actions=actions,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
elif args.player_type == 1:
    log_prefix = f"./log/run_mono_Greedy_{args.alpha}_{args.beta}_{args.gamma}_{args.horizon}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = AdaptGreedyPlayer(
        pid=0, actions=actions,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
    player_1 = AdaptGreedyPlayer(
        pid=1, actions=actions,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
elif args.player_type == 2:
    log_prefix = f"./log/run_mono_batch_{args.alpha}_{args.beta}_{args.gamma}_{args.horizon}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    player_0 = AdaptGreedyBatchPlayer(
        pid=0, actions=actions, batch_size=args.batch_size, 
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
    player_1 = AdaptGreedyBatchPlayer(
        pid=1, actions=actions, batch_size=args.batch_size, 
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
else:
    assert False, "Invalid player type."
# **************************************


# Game simulator.
# ======================================
game = MonopolyGame(
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
        r_init += game._reward_func([a,b])[0]
    r_init = r_init / (1-args.gamma) / len(actions)
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

runner = MonopolyTrajectoryRunner(game)
runner.run(
    T = args.T,
    clear_size = 10*args.batch_size,
    log_freq = args.log_freq,
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