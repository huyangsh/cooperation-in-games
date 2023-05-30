import numpy as np
from tqdm import tqdm

from game.game import Game
from game.utils import mean, _print_redirect

class MatrixGame(Game):
    def __init__(self, players, reward_matrix):
        super(__class__, self).__init__(players=players, init_state=None)
        self.reward_matrix = np.array(reward_matrix)
    
    def reward_func(self, actions):
        return self.reward_matrix[actions[0], actions[1]]
    
    def transit_func(self, state, actions):
        return tuple(actions)
    
    def _init_log(self, log_url):
        self.log = {}
        for i in range(self.n):
            self.log[f"reward_{i}"] = []
        
        f = open(log_url, "a") if log_url != "" else None
        _print_redirect(f, "GAME: Prisoner's Dilemma")
        for player in self.players:
            _print_redirect(f, player.setting)
        
        tqdm.write("\n")
        f.write("\n\n")
        if not f: f.close()
    
    def _update_log(self, t):
        for i in range(self.n):
            self.log[f"reward_{i}"].append(self.history[-1][2][i])

    def _print_log(self, t, log_url):
        f = open(log_url, "a") if log_url != "" else None

        _print_redirect(f, "\n"+"*"*30+"\n*"+f"CURRENT STEP: {t+1}".center(28)+"*\n"+"*"*30+"\n")
        
        for player in self.players:
            _print_redirect(f, player.get_print_data())

        _print_redirect(f, f"avg_reward = ({mean(self.log['reward_0']):.4f}, {mean(self.log['reward_1']):.4f})")   
        msg = "last actions ="
        for i in range(10):
            msg += f" ({self.history[-i][1][0]},{self.history[-i][1][1]}),"
        msg = msg[:-1] + "."
        _print_redirect(f, msg)

        tqdm.write("\n")
        f.write("\n\n")
        if not f: f.close()
    
    def get_data(self, t, period, full):
        if full:
            return self.log
        else:
            data = {}
            for i in range(self.n):
                data[f"reward_{i}"] = self.log[f"reward_{i}"][-period:]
            return data