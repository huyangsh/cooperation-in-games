from collections import defaultdict
from copy import deepcopy
import random
import numpy as np
from math import exp

import torch
import torch.nn as nn
import torch.optim as optim

from player.player import Player
from player.utils import LSTMQNetwork

class LSTMPlayer(Player):
    def __init__(self, pid, actions, alpha, beta, gamma, horizon, hidden_size, num_layers, lr, lr_decay, device, log_freq=0):
        self.pid = pid

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.horizon = horizon

        self.actions = actions
        self.num_actions = len(self.actions)

        self.Q_table = defaultdict(lambda: np.zeros(shape=self.num_actions, dtype=np.float32))

        self.setting = f"PLAYER {pid}: LSTMPlayer\n"
        self.setting += f"  alpha = {alpha}, beta = {beta}, gamma = {gamma}, horizon = {horizon},\n"
        self.setting += f"  #actions = {self.num_actions}, actions = {self.actions},\n"
        self.setting += f"  hidden_size = {hidden_size}, num_layers = {num_layers}, lr = {lr}, lr_decay = {lr_decay},\n"
        self.setting += f"  device = {device}, log_freq = {log_freq}."

        self.log_freq = log_freq
        self.log = {
            "Q_table": [],
            "loss": []
        }

        # LSTM-based network for predicting Q function.
        self.device = device
        self.model = LSTMQNetwork(
            input_size  = self.num_actions*2,
            hidden_size = hidden_size,
            proj_size   = 0,  # equal to hidden size.
            num_layers  = num_layers,
            output_size = self.num_actions
        ).to(self.device)

        self.loss_func = nn.MSELoss()
        self.loss_tot  = 0
        self.loss_cnt  = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=lr_decay)
        

        # Internal states.
        self.is_random = True
    
    def get_action(self, i):
        return self.actions[i]
    
    def forward(self, t, get_history):
        # LSTM forward pass: predict Q_table from history.
        n_seq = min(t-1, self.horizon)
        seq = torch.zeros(size=(n_seq, 2*self.num_actions), dtype=torch.float32).to(self.device)
        for i in range(n_seq):
            t_i = t - n_seq + i
            s_i = get_history(t_i)[0]
            seq[i, s_i[self.pid]] = 1.0
            seq[i, self.num_actions + s_i[1-self.pid]] = 1.0
        return self.model(seq)

    def play(self, t, state, get_history):
        if t <= 2:
            action = random.choice(range(self.num_actions))
        else:    
            # Take the eps-greedy action.
            eps = max(exp(-self.beta*t), 0.01)
            if (random.random() < eps):
                action = random.choice(range(self.num_actions))
                self.is_random = True
            else:
                Q_pred = self.forward(t, get_history)
                action = Q_pred.argmax().cpu().detach().item()
                self.is_random = False
        
        if (self.log_freq > 0) and ((t+1) % self.log_freq == 0):
            self._update_log()

        return action

    def update(self, t, state, get_history):
        if t > 2:
            # Update Q_table using the last step.
            prv_state = get_history(-1)[0]
            cur_state = state
            prv_action = get_history(-1)[1][self.pid]
            prv_reward = get_history(-1)[2][self.pid]
            self.Q_table[prv_state][prv_action] += self.alpha * (prv_reward + self.gamma * self.Q_table[cur_state].max() - self.Q_table[prv_state][prv_action])

            # LSTM backward pass: update the LSTM to match with the Q-function.
            self.model.zero_grad()
            Q_pred = self.forward(t, get_history).to(self.device)
            Q_target = torch.tensor(self.Q_table[prv_state], dtype=torch.float32).to(self.device)
            loss = self.loss_func(Q_pred, Q_target)

            self.loss_tot += loss.cpu().detach().item()
            self.loss_cnt += 1

            loss.backward()
            self.optimizer.step()

    def get_data(self, t, period, full):
        if self.log_freq == 0:
            return None
        
        if full:
            return (self.log["Q_table"], self.log["loss"])
        else:
            idx_start = (t+1-period) // self.log_freq
            return (self.log["Q_table"][idx_start:], self.log["loss"][idx_start:])
    
    def get_print_data(self):
        msg = ""

        msg += f"P{self.pid} |".rjust(6)
        for a1 in range(self.num_actions):
            msg += f"{a1}".rjust(4)
        msg += "\n"

        msg += "------" + " ---"*self.num_actions + "\n"
        
        for a0 in range(self.num_actions):
            msg += f"{a0} |".rjust(6)
            for a1 in range(self.num_actions):
                msg += f"{self._argmax(self.Q_table[(a0,a1)])}".rjust(4)
            msg += "\n"
        msg += "="*(6+4*self.num_actions)

        if self.loss_cnt > 10:
            msg += f"\nrecent_avg_loss = {self.loss_tot/self.loss_cnt:.4f}\n"
        elif self.log['loss']:
            msg += f"\nrecent_avg_loss = {self.log['loss'][-1]}\n"

        return msg
    
    def _update_log(self):
        self.log["Q_table"].append(deepcopy(dict(self.Q_table)))

        self.log["loss"].append(self.loss_tot/self.loss_cnt)
        self.loss_tot = 0
        self.loss_cnt = 0
    
    def _argmax(self, Q_vector):
        actions = np.argwhere(Q_vector == Q_vector.max()).flatten().tolist()
        return random.choice(actions)