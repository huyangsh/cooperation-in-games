from abc import abstractmethod
from tqdm import tqdm
import pickle as pkl
from copy import deepcopy

from game.utils import History

class Game:
    def __init__(self, players, init_state=None):
        self.players = players
        self.n = len(players)
        self.init_state = init_state

        self.reset()
    
    def reset(self):
        self.state = self.init_state
        return self.state
    
    def step(self, actions):
        reward = self._reward_func(actions)
        self.state = self._transit_func(self.state, actions)
        return reward, self.state
    
    @abstractmethod
    def _reward_func(self, actions):
        raise NotImplementedError
    
    @abstractmethod
    def _transit_func(self, state, actions):
        raise NotImplementedError


class Runner:
    def __init__(self, game):
        self.game = game
    
    @abstractmethod
    def run(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_data(self, t, period, full):
        raise NotImplementedError
    
    @abstractmethod
    def _init_log(self):
        raise NotImplementedError
    
    @abstractmethod
    def _update_log(self, t):
        raise NotImplementedError

    @abstractmethod
    def _print_log(self, t):
        raise NotImplementedError
    
    @abstractmethod
    def _save_run(self, t, period, full=False):
        raise NotImplementedError


class TrajectoryRunner(Runner):
    def __init__(self, game):
        super(__class__, self).__init__(game=game)
        self.game_eval = deepcopy(game)

    def run(self):
        if self.clear_size > 0:
            self.history = History(clear_size=self.clear_size, save_url=self.log_prefix+"_history.pkl")
        elif self.clear_size == 0:
            self.history = []
        else:
            assert False, "Clearing size should be a non-negative integer."
        self._init_log()

        state = self.game.reset()
        bar = tqdm(range(self.T))
        for t in bar:
            actions = []
            for player in self.game.players:
                player.update(t, state, self.history)
                actions.append(player.play(t, state, self.history))

            rewards, next_state = self.game.step(actions)
            self.history.append((state, actions, rewards, t))

            if self.log_freq > 0:
                self._update_log(t)
                if ((t+1) % self.log_freq == 0):
                    self._save_run(t, self.log_freq)
                    self._print_log(t)
            
            state = next_state

        if self.save_url != "":
            self._save_run(t, self.T, full=True)
            print(f"Succesfully saved to <{self.save_url}>.")
    
    def eval(self, T_eval):
        state = self.game_eval.reset()
        eval_history = []
        for t in range(self.T_eval):
            actions = []
            for player in self.game.players:
                actions.append(player.play_eval(t, state, eval_history))

            rewards, next_state = self.game_eval.step(actions)
            eval_history.append((state, actions, rewards, t))
            state = next_state
        return eval_history
    
    def _save_run(self, t, period, full=False):
        data = {}
        data["t"] = t
        
        data["simulation"] = self.get_data(t, period, full)
        for player in self.game.players:
            data[f"player_{player.pid}"] = player.get_data(t, period, full) 
        
        with open(self.save_url, "ab") as f:
            pkl.dump(data, f)


class OracleRunner(Runner):
    def __init__(self, game, random_init_func):
        super(__class__, self).__init__(game=game)
        self.game_eval = deepcopy(game)
        self.random_init_func = random_init_func

    def run(self):
        if self.clear_size > 0:
            self.history = History(clear_size=self.clear_size, save_url=self.log_prefix+"_history.pkl")
        elif self.clear_size == 0:
            self.history = []
        else:
            assert False, "Clearing size should be a non-negative integer."
        self._init_log()

        bar = tqdm(range(self.T))
        for t in bar:
            self.game.state = self.random_init_func()
            state = self.game.state

            actions = []
            for player in self.game.players:
                player.update(t, state, self.history)
                actions.append(player.play(t, state, self.history))

            rewards, next_state = self.game.step(actions)
            self.history.append((state, actions, rewards, t))

            if self.log_freq > 0:
                self._update_log(t)
                if ((t+1) % self.log_freq == 0):
                    self._save_run(t, self.log_freq)
                    self._print_log(t)

        if self.save_url != "":
            self._save_run(t, self.T, full=True)
            print(f"Succesfully saved to <{self.save_url}>.")
    
    def eval(self, T_eval):
        state = self.game_eval.reset()
        eval_history = []
        for t in range(self.T_eval):
            actions = []
            for player in self.game.players:
                actions.append(player.play_eval(t, state, eval_history))

            rewards, next_state = self.game_eval.step(actions)
            eval_history.append((state, actions, rewards, t))
            state = next_state
        return eval_history
    
    def _save_run(self, t, period, full=False):
        data = {}
        data["t"] = t
        
        data["simulation"] = self.get_data(t, period, full)
        for player in self.game.players:
            data[f"player_{player.pid}"] = player.get_data(t, period, full) 
        
        with open(self.save_url, "ab") as f:
            pkl.dump(data, f)