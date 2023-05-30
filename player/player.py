from abc import abstractmethod

class Player:
    @abstractmethod
    def play(self, t, state, get_history):
        raise NotImplementedError

    @abstractmethod
    def update(self, t, state, get_history):
        raise NotImplementedError
    
    @abstractmethod
    def get_action(self, i):
        raise NotImplementedError

    @abstractmethod
    def print_data(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_data(self, t, period, full):
        raise NotImplementedError