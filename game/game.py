from abc import abstractmethod
from tqdm import tqdm
import pickle as pkl

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
    def run(self, T, log_freq=0, log_url="", save_url=""):
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
    def _print_log(self, t, log_url):
        raise NotImplementedError
    
    @abstractmethod
    def _save_run(self, t, period, save_url, full=False):
        raise NotImplementedError


class TrajectoryRunner(Runner):
    def __init__(self, game):
        super(__class__, self).__init__(game=game)

    def run(self, T, clear_size, log_freq=0, log_url="", save_url=""):
        if clear_size > 0:
            self.history = History(clear_size=clear_size)
        elif clear_size == 0:
            self.hisotry = []
        else:
            assert False, "Clearing size should be a non-negative integer."
        self._init_log(log_url)

        state = self.game.reset()
        bar = tqdm(range(T))
        for t in bar:
            actions = []
            for player in self.game.players:
                player.update(t, state, self.history)
                actions.append(player.play(t, state, self.history))

            rewards, next_state = self.game.step(actions)
            self.history.append((state, actions, rewards, t))

            if log_freq > 0:
                self._update_log(t)
                if ((t+1) % log_freq == 0):
                    self._print_log(t, log_url)
                    self._save_run(t, log_freq, save_url)
            
            state = next_state

        if save_url != "":
            self._save_run(t, T, save_url, full=True)
            print(f"Succesfully saved to <{save_url}>.")
    
    def _save_run(self, t, period, save_url, full=False):
        data = {}
        data["t"] = t
        
        data["simulation"] = self.get_data(t, period, full)
        for player in self.game.players:
            data[f"player_{player.pid}"] = player.get_data(t, period, full) 
        
        if full:
            with open(save_url, "wb") as f:
                pkl.dump(data, f)
        else:
            with open(save_url, "ab") as f:
                pkl.dump(data, f)