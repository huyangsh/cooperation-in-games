import numpy as np
from math import exp

# Configuring action space.
# ======================================
M       = 15
XI      = 0.1
PN      = 1.61338
PM      = 1.73153

PN_     = 1.61169214
actions = np.linspace(PN - XI*(PM-PN), PM + XI*(PM-PN), num=M)
actions = np.hstack([
    np.linspace(PN - XI*(PM-PN), PN_, num=50)[:-1],
    np.linspace(PN_, PM + XI*(PM-PN), num=M-1)
])
print(actions)

n = 2
aa = [2, 2]
a0 = 1
mu = 0.5
c = [1, 1]
share0 = exp(a0 / mu)
def _reward_func(act):
    prices = [actions[act[i]] for i in range(n)]
    share = [exp((aa[i] - prices[i]) / mu) for i in range(n)]
    total_share = sum(share) + share0
    demand = [share[i] / total_share for i in range(n)]
    return [(prices[i] - c[i]) * demand[i] for i in range(n)]
gamma = 0.95

init_values = []
for a in range(len(actions)):
    r_init = 0
    for b in range(len(actions)):
        r_init += _reward_func([a,b])[0]
    r_init = r_init / (1-gamma) / len(actions)
    init_values.append(r_init)
print(init_values)