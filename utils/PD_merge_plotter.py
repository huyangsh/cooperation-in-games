import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import itertools

HORIZON  = 1
LOG_FREQ = 50000
actions = np.array([0, 1])
states = list(itertools.product(actions, repeat=2))

'''log_prefixes = [
    "./log/PD_LSTM/run_PD_LSTM_0.1_1_16_1_20230502_200244",
    "./log/PD_LSTM/run_PD_LSTM_0.1_2_16_1_20230502_200244",
    "./log/PD_LSTM/run_PD_LSTM_0.1_4_16_1_20230502_200302",
    "./log/PD_LSTM/run_PD_LSTM_0.1_8_16_1_20230502_200312",
    "./log/PD_LSTM/run_PD_LSTM_0.1_16_16_1_20230502_200317",
]'''
'''log_prefixes = [
    "./log/PD_LSTM/run_PD_LSTM_0.1_1_16_2_20230502_184426",
    "./log/PD_LSTM/run_PD_LSTM_0.1_2_16_2_20230502_184426",
    "./log/PD_LSTM/run_PD_LSTM_0.1_4_16_2_20230502_184432",
    "./log/PD_LSTM/run_PD_LSTM_0.1_8_16_2_20230502_184432",
    "./log/PD_LSTM/run_PD_LSTM_0.1_16_16_2_20230502_184433",
]'''
'''log_prefixes = [
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_1_16_1_32_20230502_193732",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_2_16_1_32_20230502_193732",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_4_16_1_32_20230502_193741",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_8_16_1_32_20230502_193741",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_16_16_1_32_20230502_194051",
]'''
'''log_prefixes = [
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_1_16_2_32_20230502_184738",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_2_16_2_32_20230502_184806",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_4_16_2_32_20230502_184823",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_8_16_2_32_20230502_184823",
    "./log/PD_LSTM/run_PD_LSTM_Batch_0.1_16_16_2_32_20230502_184823",
]'''
log_prefixes = []
n = len(log_prefixes)

fig = plt.figure(figsize=(20,30))
fig.subplots_adjust(hspace=0.25)
for k in range(n):
    with open(log_prefixes[k] + ".pkl", "rb") as f:
        data = pkl.load(f)

    # Plot Q-table.
    x = np.array(range(len(data["player_0"]))) * LOG_FREQ

    i = 0
    for s in states:
        i = i + 1

        ax = fig.add_subplot(2*n,4,k*8+i)
        y_0 = np.array([x[s][0] for x in data["player_0"]])
        y_1 = np.array([x[s][1] for x in data["player_0"]])
        ax.plot(x, y_0, label="C")
        ax.plot(x, y_1, label="D")
        ax.legend(loc="lower left")
        if i == 1: ax.set_ylabel("Player 0")

        ax = fig.add_subplot(2*n,4,k*8+4+i)
        y_0 = np.array([x[s][0] for x in data["player_1"]])
        y_1 = np.array([x[s][1] for x in data["player_1"]])
        ax.plot(x, y_0, label="C")
        ax.plot(x, y_1, label="D")
        ax.legend(loc="lower left")
        ax.set_xlabel(f"state = {s}")
        if i == 1: ax.set_ylabel("Player 1")
fig.savefig("./utils/PD_merge_Q-table.png", dpi=200)