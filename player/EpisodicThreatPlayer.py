from collections import defaultdict
import numpy as np
import random

from player.player import Player

class EpisodicThreatPlayer(Player):
    def __init__(self, pid, C_id, C_action, D_id, D_action, period):
        self.pid = pid

        self.C_id     = C_id
        self.C_action = C_action
        self.D_id     = D_id
        self.D_action = D_action

        self.period = period

        self.setting = f"PLAYER {pid}: TitForTatPlayer\n"
        self.setting += f"  C_action = (#{self.C_id}, {self.C_action}), D_action = (#{self.D_id}, {self.D_action}), period = {self.period}."

        # Internal states.
        self.threat = False
    
    def get_action(self, i):
        if i == self.C_id:
            return self.C_action
        elif i == self.D_id:
            return self.D_action
        else:
            assert False, f"Invalid action {i} for a ThreatPlayer instance."

    def play(self, t, state, get_history):
        if (t % self.period == 0):
            self.threaten = False
            return self.C_id
        else:
            if state[1-self.pid] == D_id:
                self.threaten = True
            
            return self.D_id if self.threaten else self.C_id

    def update(self, t, state, get_history):
        pass
        
    def get_data(self, t, period, full):
        return None
    
    def get_print_data(self):
        return ""