import numpy as np
from math import exp
from tqdm import tqdm

from game.game import Game
from game.utils import mean, _print_redirect

class MonopolyGame(Game):
    def __init__(self, players, a, a0, mu, c):
        super(__class__, self).__init__(players=players, init_state=None)
        self.a = a
        self.mu = mu
        self.c = c
        self.share0 = exp(a0 / mu)
    
    def reward_func(self, actions):
        prices = [self.players[i].get_action(actions[i]) for i in range(self.n)]
        share = [exp((self.a[i] - prices[i]) / self.mu) for i in range(self.n)]
        total_share = sum(share) + self.share0
        demand = [share[i] / total_share for i in range(self.n)]
        return [(prices[i] - self.c[i]) * demand[i] for i in range(self.n)]

    def transit_func(self, state, actions):
        return tuple(actions)
    
    def _init_log(self, log_url):
        self.log = {}
        for i in range(self.n):
            self.log[f"reward_{i}"] = []
        self.log["visit"] = np.zeros(shape=(self.players[0].num_actions, self.players[1].num_actions))
        self.log["explore"] = np.zeros(shape=(self.players[0].num_actions, self.players[1].num_actions))

        f = open(log_url, "a") if log_url != "" else None
        _print_redirect(f, "GAME: Bertrand's monopoly")
        for player in self.players:
            _print_redirect(f, player.setting)
        
        tqdm.write("\n")
        f.write("\n\n")
        if not f: f.close()
    
    def _update_log(self, t):
        for i in range(self.n):
            self.log[f"reward_{i}"].append(self.history[-1][2][i])
        self.log["visit"][self.history[-1][1][0],self.history[-1][1][1]] += 1

        if self.players[0].is_random or self.players[1].is_random:
            self.log["explore"][self.history[-1][1][0],self.history[-1][1][1]] += 1
    
    def _print_log(self, t, log_url):
        f = open(log_url, "a") if log_url != "" else None

        _print_redirect(f, "\n"+"*"*30+"\n*"+f"CURRENT STEP: {t+1}".center(28)+"*\n"+"*"*30+"\n")
        
        for player in self.players:
            _print_redirect(f, player.get_print_data())
        
        _print_redirect(f, self._get_visit_str(visit_count=self.log["visit"], title="s_t"))
        _print_redirect(f, self._get_visit_str(visit_count=self.log["explore"], title="exp"))
        _print_redirect(f, f"#exploration = {int(self.log['explore'].sum())}\n")

        _print_redirect(f, f"avg_reward = ({mean(self.log['reward_0']):.4f}, {mean(self.log['reward_1']):.4f})")
        # tqdm.write(f"avg_last_100_rewards = ({mean(self.log['reward_0'][:-100]):.4f}, {mean(self.log['reward_1'][:-100]):.4f})")
        
        msg = "last actions ="
        for i in range(10):
            msg += f" ({self.history[-i][1][0]},{self.history[-i][1][1]}),"
        msg = msg[:-1] + "."
        _print_redirect(f, msg)

        tqdm.write("\n")
        f.write("\n\n")
        if not f: f.close()

        self.log["visit"] = np.zeros(shape=(self.players[0].num_actions, self.players[1].num_actions))
        self.log["explore"] = np.zeros(shape=(self.players[0].num_actions, self.players[1].num_actions))
    
    def _get_visit_str(self, visit_count, title):
        msg = ""
        tot_visit = visit_count.sum()
        if tot_visit > 0:
            visit_freq = visit_count / visit_count.sum() * 10000
        else:
            visit_freq = visit_count

        msg += f"{title} |".rjust(6)
        for a1 in range(self.players[1].num_actions):
            msg += f"{a1}".rjust(6)
        msg += "\n"

        msg += "------" + " -----"*self.players[1].num_actions + "\n"
        
        for a0 in range(self.players[0].num_actions):
            msg += f"{a0} |".rjust(6)
            for a1 in range(self.players[1].num_actions):
                msg += f"{int(visit_freq[a0,a1])}".rjust(6)
            msg += "\n"
        msg += "="*(6+6*self.players[1].num_actions)

        return msg

    def get_data(self, t, period, full):
        if full:
            return self.log
        else:
            data = {}
            for i in range(self.n):
                data[f"reward_{i}"] = self.log[f"reward_{i}"][-period:]
            return data