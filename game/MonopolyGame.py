import numpy as np
from math import exp
from tqdm import tqdm
from copy import deepcopy

from game.game import Game, TrajectoryRunner
from game.utils import mean, logging

class MonopolyGame(Game):
    def __init__(self, players, a, a0, mu, c):
        super(__class__, self).__init__(players=players, init_state=None)
        self.a = a
        self.mu = mu
        self.c = c
        self.share0 = exp(a0 / mu)
    
    def _reward_func(self, actions):
        prices = [self.players[i].get_action(actions[i]) for i in range(self.n)]
        share = [exp((self.a[i] - prices[i]) / self.mu) for i in range(self.n)]
        total_share = sum(share) + self.share0
        demand = [share[i] / total_share for i in range(self.n)]
        return [(prices[i] - self.c[i]) * demand[i] for i in range(self.n)]

    def _transit_func(self, state, actions):
        return tuple(actions)


class MonopolyTrajectoryRunner(TrajectoryRunner):
    def __init__(self, game, T, clear_size, log_freq=0, log_prefix="", T_eval=0):
        super(__class__, self).__init__(game=game)
        self.T = T
        self.T_eval = T_eval
        self.clear_size = clear_size

        self.log_freq = log_freq

        self.log_prefix = log_prefix
        self.log_url = log_prefix + ".log"
        self.save_url = log_prefix + "_run.pkl"
    
    def _init_log(self):
        self.log = {}
        self.log["traj_rewards"] = []
        self.log["eval_rewards"] = []
        for i in range(self.game.n):
            self.log[f'Bellman_{i}'] = []
        self.log["visit"] = np.zeros(shape=(self.game.players[0].num_actions, self.game.players[1].num_actions))
        self.log["explore"] = np.zeros(shape=(self.game.players[0].num_actions, self.game.players[1].num_actions))
    
    def _update_log(self, t):
        self.log["visit"][self.history[-1][1][0],self.history[-1][1][1]] += 1

        if self.game.players[0].is_random or self.game.players[1].is_random:
            self.log["explore"][self.history[-1][1][0],self.history[-1][1][1]] += 1
    
    def _print_log(self, t):
        f = open(self.log_url, "a") if self.log_url != "" else None

        logging(f, "\n"+"*"*30+"\n*"+f"CURRENT STEP: {t+1}".center(28)+"*\n"+"*"*30+"\n")
        
        for player in self.game.players:
            logging(f, player.get_print_data())
        
        logging(f, self._get_visit_str(visit_count=self.log["visit"], title="s_t"))
        logging(f, self._get_visit_str(visit_count=self.log["explore"], title="exp"))
        logging(f, f"#exploration = {int(self.log['explore'].sum())}\n")

        # logging(f, f"avg_reward = ({mean(self.log['reward_0']):.4f}, {mean(self.log['reward_1']):.4f})")
        msg = "avg_last_1000_rewards = ["
        avg_rewards = np.array([x[2] for x in self.history[-1000:]]).mean(axis=0)
        self.log["traj_rewards"].append(avg_rewards)
        for i in range(self.game.n):
            msg += "{:.4f}, ".format(avg_rewards[i])
        msg = msg[:-2] + "]."
        logging(f, msg)

        if self.T_eval > 0:
            eval_history = self.eval(self.T_eval)
            eval_traj = [x[0] for x in eval_history]

            cum_rewards = np.array([x[2] for x in eval_history])
            gammas = np.array([self.game.players[0].gamma, self.game.players[1].gamma])[np.newaxis, :]
            gammas = np.repeat(gammas, repeats=self.T_eval, axis=0)
            gammas = np.power(gammas, np.repeat(np.arange(self.T_eval)[:, np.newaxis], repeats=2, axis=1))
            cum_rewards = (cum_rewards*gammas).sum(axis=0)
            logging(f, f"Evaluation for {self.T_eval} steps: {eval_traj}.\neval_rewards = {cum_rewards[0]:.4f}, {cum_rewards[1]:.4f}")
        
        msg = "last actions ="
        for i in range(-10, 0, 1):
            msg += f" ({self.history[i][1][0]},{self.history[i][1][1]}),"
        msg = msg[:-1] + "."
        logging(f, msg)

        msg = "Bellman difference = ["
        for i in range(self.game.n):
            self.log[f"Bellman_{i}"].append(self._calc_Bellman_diff(i, t))
            msg += "{:.4f}, ".format(self.log[f"Bellman_{i}"][-1])
        msg = msg[:-2] + "]."
        logging(f, msg)

        tqdm.write("\n")
        f.write("\n\n")
        if f is not None: f.close()

        self.log["visit"] = np.zeros(shape=(self.game.players[0].num_actions, self.game.players[1].num_actions))
        self.log["explore"] = np.zeros(shape=(self.game.players[0].num_actions, self.game.players[1].num_actions))
    
    def _get_visit_str(self, visit_count, title):
        msg = ""
        tot_visit = visit_count.sum()
        if tot_visit > 0:
            visit_freq = visit_count / visit_count.sum() * 10000
        else:
            visit_freq = visit_count

        msg += f"{title} |".rjust(6)
        for a1 in range(self.game.players[1].num_actions):
            msg += f"{a1}".rjust(6)
        msg += "\n"

        msg += "------" + " -----"*self.game.players[1].num_actions + "\n"
        
        for a0 in range(self.game.players[0].num_actions):
            msg += f"{a0} |".rjust(6)
            for a1 in range(self.game.players[1].num_actions):
                msg += f"{int(visit_freq[a0,a1])}".rjust(6)
            msg += "\n"
        msg += "="*(6+6*self.game.players[1].num_actions)

        return msg
    
    def _calc_Bellman_diff(self, i, t):
        Q_dict = self.game.players[i].Q_table
        
        diff = 0
        for state in Q_dict.keys():
            Q_state = np.zeros(shape=(self.game.players[i].num_actions,))
            for action_i in range(self.game.players[i].num_actions):
                actions = []
                for j in range(self.game.n):
                    if j == i:
                        actions.append(Q_dict[state].argmax())
                    else:
                        actions.append(action_i)
                next_state = self.game._transit_func(state, actions)
                Q_state[actions[i]] = self.game._reward_func(actions)[i] + self.game.players[i].gamma * Q_dict[next_state].max()
            diff += np.linalg.norm(Q_dict[state] - Q_state) ** 2
        
        return np.sqrt(diff)


    def get_data(self, t, period, full):
        if full:
            return self.log
        else:
            data = {"visit": self.log["visit"], "explore": self.log["explore"]}
            return data