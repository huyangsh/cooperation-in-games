import numpy as np
import pickle as pkl

def merge_dict(dict_list):
    keys = set(dict_list[0].keys())
    for dic in dict_list:
        if len(set(dic.keys()) - keys) > 0:
            assert False, "Invalid partial log: periodic keys do not agree."
    
    merged_dict = {}
    for key in keys:
        merged_dict[key] = []
        for dic in dict_list:
            data = dic[key]
            if type(data) == list:
                merged_dict[key] += data
            else:
                merged_dict[key].append(data)
    return merged_dict

log_prefix = "./log/run_PD_0.1_0.05_0.9_1_20230502_141655"

dict_list = []
with open(log_prefix + ".pkl", "rb") as f:
    while True:
        try:
            dic = pkl.load(f)
        except EOFError:
            break
        dict_list.append(dic)


merged_dict = merge_dict(dict_list)
reward_dict = merge_dict(merged_dict["simulation"])
merged_dict["simulation"] = reward_dict

# print(merged_dict["simulation"].keys(), len(merged_dict["simulation"]["reward_0"]), len(merged_dict["simulation"]["reward_1"]))
# print(len(merged_dict["player_0"]), len(merged_dict["player_1"]), len(merged_dict["t"]))