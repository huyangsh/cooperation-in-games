import pickle as pkl

url = "./log/monopoly_AER_0.1_2e-05_0.95_1_0_20230530_162034_history.pkl"

history = []
with open(url, "rb") as f:
    while True:
        try:
            history += pkl.load(f)
        except EOFError:
            break

print(len(history))
print(history[:1000])