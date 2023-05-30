from abc import abstractmethod
from tqdm import tqdm
import pickle as pkl

class Game:
    def __init__(self, players, init_state=None):
        self.players = players
        self.n = len(players)
        self.init_state = init_state
    
    @abstractmethod
    def reward_func(self, actions):
        raise NotImplementedError
    
    @abstractmethod
    def transit_func(self, state, actions):
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
    def get_data(self, t, period, full):
        raise NotImplementedError
    
    def get_history(self, t):
        if t >= self.history_st:
            entry = self.history[t-self.history_st]
            assert entry[-1] == t
            return entry
        elif t < 0 and t >= -len(self.history):
            return self.history[t]
        else:
            assert False, "Error: requested history has been cleared."
    
    def run(self, T, clear_freq, log_freq=0, log_url="", save_url=""):
        assert clear_freq >= 0, "Clearing frequency should be a non-negative integer."

        self.history, self.history_st, state = [], 0, self.init_state
        self._init_log(log_url)

        bar = tqdm(range(T))
        for t in bar:
            actions = []
            for player in self.players:
                player.update(t, state, self.get_history)
                actions.append(player.play(t, state, self.get_history))

            rewards = self.reward_func(actions)
            self.history.append((state, actions, rewards, t))

            if log_freq > 0:
                self._update_log(t)
                if ((t+1) % log_freq == 0):
                    self._print_log(t, log_url)
                    self._save_run(t, log_freq, save_url)
            
            if (clear_freq > 0) and (t % clear_freq == 0) and (t >= self.history_st + 2*clear_freq):
                self.history_st += clear_freq
                self.history = self.history[clear_freq:]
            
            state = self.transit_func(state, actions)

        if save_url != "":
            self._save_run(t, T, save_url, full=True)
            print(f"Succesfully saved to <{save_url}>.")
    
    def _save_run(self, t, period, save_url, full=False):
        data = {}
        data["t"] = t
        
        data["simulation"] = self.get_data(t, period, full)
        for player in self.players:
            data[f"player_{player.pid}"] = player.get_data(t, period, full) 
        
        if full:
            with open(save_url, "wb") as f:
                pkl.dump(data, f)
        else:
            with open(save_url, "ab") as f:
                pkl.dump(data, f)