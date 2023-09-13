import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from itertools import product

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

actions = np.linspace(PN - XI*(PM-PN), PM + XI*(PM-PN), num=M)
"""PN_     = 1.61169214
actions = np.hstack([
    np.linspace(PN - XI*(PM-PN), PN_, num=50)[:-1],
    np.linspace(PN_, PM + XI*(PM-PN), num=M-1)
])"""

if args.player_type == 0:
    log_prefix = f"./log/monopoly_AER_{args.runner}_{args.alpha}_{args.beta}_{args.gamma}_{args.horizon}_{args.seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    log_prefix = f"./log/monopoly_Greedy_{args.runner}_{args.alpha}_{args.beta}_{args.gamma}_{args.horizon}_{args.batch_size}_{args.seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    log_prefix = f"./log/monopoly_batch_{args.runner}_{args.alpha}_{args.beta}_{args.gamma}_{args.horizon}_{args.seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

# Logging.
with open(log_prefix+".log", "a") as f:
    logging(f, "GAME: Bertrand's monopoly")
    logging(f, player_0.setting)
    logging(f, player_1.setting)
    logging(f, f"Using random seed {args.seed}.\n\n")

    # Default: Q-table initialized to Q*.
    '''logging(f, " "*5 + "|" + "".join([f"a_{a}".rjust(8) for a in range(len(actions))]))
    logging(f, "-"*6 + " -------"*len(actions))
    
    msg = "  Q* |"
    for a in range(len(actions)):
        r_init = 0
        for b in range(len(actions)):
            r_init += game._reward_func([a,b])[0]
        r_init = r_init / (1-args.gamma) / len(actions)
        msg +=  f"{r_init:.3f}".rjust(8)

        for s_0 in range(len(actions)):
            for s_1 in range(len(actions)):
                player_0.Q_table[(s_0,s_1)][a] = r_init
                player_1.Q_table[(s_0,s_1)][a] = r_init
    logging(f, msg+"\n\n")'''

    # Alternative: Q-table initialized to random.
    '''for s_0 in range(len(actions)):
        for s_1 in range(len(actions)):
            player_0.Q_table[(s_0,s_1)] = np.random.rand(len(actions)) * 3
            player_1.Q_table[(s_0,s_1)] = np.random.rand(len(actions)) * 3'''
    
    # Alternative: Q-table initialized to 0.
    '''for s_0 in range(len(actions)):
        for s_1 in range(len(actions)):
            player_0.Q_table[(s_0,s_1)] = np.zeros(shape=(len(actions),))
            player_1.Q_table[(s_0,s_1)] = np.zeros(shape=(len(actions),))'''
    
    # Alternative: Q-table initialized to single-peak.
    '''s_0_star, s_1_star = 9, 9
    Q_init_0 = np.zeros(shape=(len(actions),))
    Q_init_0[s_0_star] = 1000
    Q_init_1 = np.zeros(shape=(len(actions),))
    Q_init_1[s_1_star] = 1000
    for s_0 in range(len(actions)):
        for s_1 in range(len(actions)):
            player_0.Q_table[(s_0,s_1)] = deepcopy(Q_init_0)
            player_1.Q_table[(s_0,s_1)] = deepcopy(Q_init_1)'''
    
    states = list(product(range(len(actions)), repeat=2))
    num_states = len(states)
    def calc_Q(pi_0, pi_1, thres):
        Q = dict()
        for s in states:
            Q[s] = np.zeros(shape=(len(actions),))

        diff = thres + 1
        while diff > thres:
            Q_prev = deepcopy(Q)

            for s in states:
                a_1 = pi_1[s]
                for a_0 in range(len(actions)):
                    s_ = (a_0, a_1)
                    Q[s][a_0] = game._reward_func(s_)[0] + args.gamma * Q_prev[s_][pi_0[s_]]
            
            diff = 0
            for s in states:
                diff += np.linalg.norm(Q[s] - Q_prev[s]) ** 2
            diff = np.sqrt(diff)
        
        return Q
    
    pi_0, pi_1 = dict(), dict()
    for s in states:
        pi_0[s] = 9 if s[1] >= 9 else 1
        pi_1[s] = 9 if s[0] >= 9 else 1
    
    Q_init_0 = calc_Q(pi_0, pi_1, 0.001)
    Q_init_1 = dict()
    for s in states:
        Q_init_1[s] = Q_init_0[(s[1], s[0])]

    player_0.Q_table = deepcopy(Q_init_0)
    player_1.Q_table = deepcopy(Q_init_1)


if True:
    state = game.reset()
    eval_history = []

    T_eval = 1000
    for t in range(T_eval):
        actions_ = []
        for player in game.players:
            actions_.append(player.play_eval(t, state, eval_history))

        rewards, next_state = game.step(actions_)
        eval_history.append((state, actions_, rewards, t))
        state = next_state
    
    eval_traj = [x[0] for x in eval_history]
    cum_rewards = np.array([x[2] for x in eval_history])
    gammas = np.array([game.players[0].gamma, game.players[1].gamma])[np.newaxis, :]
    gammas = np.repeat(gammas, repeats=T_eval, axis=0)
    gammas = np.power(gammas, np.repeat(np.arange(T_eval)[:, np.newaxis], repeats=2, axis=1))
    cum_rewards = (cum_rewards*gammas).sum(axis=0)
    print(f"Evaluation for {T_eval} steps: {eval_traj}.\neval_rewards = {cum_rewards[0]:.4f}, {cum_rewards[1]:.4f}")


for it in tqdm(range(100)):
    # Fix Player 1, update Player 0.
    for t in range(100):
        new_Q_table = deepcopy(dict(player_0.Q_table))
        for s_0 in range(len(actions)):
            for s_1 in range(len(actions)):
                for a in range(len(actions)):
                    state = (s_0, s_1)
                    action = (a, player_1.play_eval(0, state, None))
                    new_Q_table[state][a] = game._reward_func(action)[0] + args.gamma*player_0.Q_table[action].max()
        player_0.Q_table = new_Q_table
    
    # Fix Player 0, update Player 1.
    for t in range(100):
        new_Q_table = deepcopy(dict(player_1.Q_table))
        for s_0 in range(len(actions)):
            for s_1 in range(len(actions)):
                for a in range(len(actions)):
                    state = (s_0, s_1)
                    action = (player_0.play_eval(0, state, None), a)
                    new_Q_table[state][a] = game._reward_func(action)[1] + args.gamma*player_1.Q_table[action].max()
        player_1.Q_table = new_Q_table
    
    # Evaluation
    if (it+1) % 1 == 0:
        state = game.reset()
        eval_history = []

        T_eval = 1000
        for t in range(T_eval):
            actions_ = []
            for player in game.players:
                actions_.append(player.play_eval(t, state, eval_history))

            rewards, next_state = game.step(actions_)
            eval_history.append((state, actions_, rewards, t))
            state = next_state
        
        eval_traj = [x[0] for x in eval_history]
        cum_rewards = np.array([x[2] for x in eval_history])
        gammas = np.array([game.players[0].gamma, game.players[1].gamma])[np.newaxis, :]
        gammas = np.repeat(gammas, repeats=T_eval, axis=0)
        gammas = np.power(gammas, np.repeat(np.arange(T_eval)[:, np.newaxis], repeats=2, axis=1))
        cum_rewards = (cum_rewards*gammas).sum(axis=0)
        print(f"Evaluation for {T_eval} steps: {eval_traj}.\neval_rewards = {cum_rewards[0]:.4f}, {cum_rewards[1]:.4f}")
# **************************************


# Plotting.
# ======================================