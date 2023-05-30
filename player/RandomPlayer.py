from collections import defaultdict
import numpy as np
import random

from player.player import Player

class RandomPlayer(Player):
    def __init__(self, pid, actions, prob):
        self.pid = pid

        self.actions = actions
        self.num_actions = len(self.actions)
        
        self.prob = prob
        self.prob /= self.prob.sum()

        self.setting = f"PLAYER {pid}: RandomPlayer\n"
        self.setting += f"  #actions = {self.num_actions}, actions = {self.actions},\n"
        self.setting += f"  prob = {self.prob}."
    
    def get_action(self, i):
        return self.actions[i]

    def play(self, t, state, history):
        return random.choices(range(self.num_actions), weights=self.prob, k=1)[0]

    def update(self, t, state, history):
        pass
        
    def get_data(self, t, period, full):
        return None
    
    def get_print_data(self):
        return ""