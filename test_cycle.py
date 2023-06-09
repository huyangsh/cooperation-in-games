import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
from tqdm import tqdm
import pickle as pkl

from game import MonopolyGame, MonopolyTrajectoryRunner, MonopolyOracleRunner
from player import AERPlayer, AdaptGreedyPlayer, AdaptGreedyBatchPlayer
from game.utils import logging


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
parser.add_argument("--T_eval", default=0, type=int)
parser.add_argument("--log_freq", default=int(5e5), type=int)
parser.add_argument("--clear_size", default=int(1e5), type=int)

parser.add_argument("--runner", default="trajectory", type=str, choices=["trajectory", "oracle"])
parser.add_argument("--draw_Q_table", action="store_true")

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
# **************************************


# Players and action space.
# ======================================
M       = 15
XI      = 0.1
PN      = 1.61338
PM      = 1.73153

action_list = np.linspace(PN - XI*(PM-PN), PM + XI*(PM-PN), num=M)
"""PN_     = 1.61169214
action_list = np.hstack([
    np.linspace(PN - XI*(PM-PN), PN_, num=50)[:-1],
    np.linspace(PN_, PM + XI*(PM-PN), num=M-1)
])"""

if args.player_type == 0:
    player_0 = AERPlayer(
        pid=0, actions=action_list,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
    player_1 = AERPlayer(
        pid=1, actions=action_list,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
elif args.player_type == 1:
    player_0 = AdaptGreedyPlayer(
        pid=0, actions=action_list,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
    player_1 = AdaptGreedyPlayer(
        pid=1, actions=action_list,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
elif args.player_type == 2:
    player_0 = AdaptGreedyBatchPlayer(
        pid=0, actions=action_list, batch_size=args.batch_size, 
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
    player_1 = AdaptGreedyBatchPlayer(
        pid=1, actions=action_list, batch_size=args.batch_size, 
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, horizon=args.horizon,
        log_freq=args.log_freq,
    )
else:
    assert False, "Invalid player type."
# **************************************


# Game simulator.
# ======================================
path = "./log/AER_15-actions/monopoly_AER_0.1_2e-05_0.95_1_50_20230601_201935_run.pkl"
with open(path, "rb") as f:
    while True:
        try:
            data = pkl.load(f)
        except EOFError:
            break
    
    player_0.Q_table = data["player_0"][-1]
    player_1.Q_table = data["player_1"][-1]

game = MonopolyGame(
    players = [player_0, player_1],
    a = [2, 2],
    a0 = 1,
    mu = 0.5,
    c = [1, 1]
)

graph, nodes = {}, []
for a_0 in range(len(action_list)):
    for a_1 in range(len(action_list)):
        state = (a_0, a_1)
        game.state = state
        graph[state] = (player_0.play_eval(0, state, None), player_1.play_eval(0, state, None))
        nodes.append(state)
    
nodes = set(nodes)
on_traj = []
while len(nodes) > 0:
    state = nodes.pop()
    path = [state]
    state = graph[state]
    while state in nodes:
        nodes.remove(state)
        path.append(state)
        state = graph[state]
    
    if state in path:
        pos = path.index(state)
        print(path[pos:])
        on_traj += path[pos:]

for state in on_traj:
    game.state = state
    print(state, end=": ")

    a_correct = player_0.play_eval(0, state, None)
    cum_reward_list = []
    for a_ in range(len(action_list)):
        actions = [a_, player_1.play_eval(0, state, None)]
        rewards, state = game.step(actions)

        cum_reward, coeff = rewards[0], args.gamma
        for t in range(1,1000):
            actions = [player_0.play_eval(t, state, None), player_1.play_eval(t, state, None)]
            rewards, state = game.step(actions)
            cum_reward += coeff*rewards[0]
            coeff *= args.gamma
        
        cum_reward_list.append(cum_reward)
        if a_ == a_correct:
            print(f"({cum_reward}), ", end="")
        else:
            print(f"{cum_reward}, ", end="")
    
    if max(cum_reward_list) == cum_reward_list[a_correct]:
        print("OK")
    else:
        print("Not Nash.")
    print()
# ======================================
