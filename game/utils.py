from tqdm import tqdm

def mean(data):
    return sum(data) / len(data)

def _print_redirect(f, msg):
    tqdm.write(msg)
    if f: f.write(msg+"\n")


class History:
    def __init__(self, clear_size):
        self.clear_size = clear_size

        self.data = []
        self.ptr_st, self.ptr_ed = 0, -1
    
    def __getitem__(self, t):
        if type(t) == int:
            if t >= self.ptr_st and t <= self.ptr_ed:
                return self.data[t-self.ptr_st]
            elif t < 0 and t >= -len(self.data):
                return self.data[t]
            else:
                assert False, "Requested history has been cleared."
        elif type(t) == slice:
            # The implementation is problematic: when start is omitted, regarded as 0.
            start, stop, step = t.indices(self.ptr_ed+1)
            # assert start >= self.ptr_st and stop >= start and stop <= self.ptr_ed+1
            if stop <= self.ptr_ed:
                return self.data[start-self.ptr_st:stop-self.ptr_st:step]
            else:
                return self.data[start-self.ptr_st::step]
        else:
            raise NotImplementedError
    
    def append(self, x):
        self.data.append(x)
        self.ptr_ed += 1

        length = self.ptr_ed - self.ptr_st + 1
        assert length == len(self.data)

        if (self.clear_size > 0) and (length >= 2*self.clear_size):
            self.ptr_st = self.ptr_ed - self.clear_size + 1
            self.data = self.data[-self.clear_size:]