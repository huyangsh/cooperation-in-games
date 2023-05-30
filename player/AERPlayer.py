from collections import defaultdict
from copy import deepcopy
import random
import numpy as np
from math import exp

from player.player import Player

class AERPlayer(Player):
    def __init__(self, pid, actions, alpha, beta, gamma, horizon, log_freq=0):
        self.pid = pid

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.horizon = horizon

        self.actions = actions
        self.num_actions = len(self.actions)

        self.Q_table = defaultdict(lambda: np.zeros(shape=self.num_actions, dtype=np.float32))

        self.setting = f"PLAYER {pid}: AERPlayer\n"
        self.setting += f"  alpha = {alpha}, beta = {beta}, gamma = {gamma}, horizon = {horizon},\n"
        self.setting += f"  #actions = {self.num_actions}, actions = {self.actions}."

        self.log_freq = log_freq
        self.log = {
            "Q_table": []
        }

        # Internal states.
        self.is_random = True
    
    def get_action(self, i):
        return self.actions[i]

    def play(self, t, state, get_history):
        if t <= 2:
            action = random.choice(range(self.num_actions))
        else:    
            # Take the eps-greedy action.
            eps = exp(-self.beta*t)
            if (random.random() < eps):
                action = random.choice(range(self.num_actions))
                self.is_random = True
            else:
                action = self._argmax(self.Q_table[state])
                self.is_random = False
        
        if (self.log_freq > 0) and ((t+1) % self.log_freq == 0):
            self._update_log()

        return action

    def update(self, t, state, get_history):
        if t > 2:
            # Update using the last step.
            prv_state = get_history(-1)[0]
            cur_state = state
            prv_action = get_history(-1)[1][self.pid]
            prv_reward = get_history(-1)[2][self.pid]
            self.Q_table[prv_state][prv_action] += self.alpha * (prv_reward + self.gamma * self.Q_table[cur_state].max() - self.Q_table[prv_state][prv_action])

    def get_data(self, t, period, full):
        if self.log_freq == 0:
            return None
        
        if full:
            return self.log["Q_table"]
        else:
            idx_start = (t+1-period) // self.log_freq
            return self.log["Q_table"][idx_start:]
    
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

        return msg
    
    def _update_log(self):
        self.log["Q_table"].append(deepcopy(dict(self.Q_table)))
    
    def _argmax(self, Q_vector):
        actions = np.argwhere(Q_vector == Q_vector.max()).flatten().tolist()
        return random.choice(actions)