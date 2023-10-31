# Standard Library
import gc

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score




# First Party Library
import csv
import config
from env import Env
from init_real_data import init_real_data

#another file
from memory import Memory
from actor import Actor
from critic import Critic

device = config.select_device






torch.autograd.set_detect_anomaly(True)
#alpha,betaの読み込み
np_alpha = []
np_beta = []
with open("model.param.data.fast", "r") as f:
    lines = f.readlines()
    for line in lines:
        datas = line[:-1].split(",")
        np_alpha.append(np.float32(datas[0]))
        np_beta.append(np.float32(datas[1]))

T = np.array(
    [0.8 for i in range(len(np_alpha))],
    dtype=np.float32,
)
e = np.array(
    [0.8 for i in range(len(np_beta))],
    dtype=np.float32,
)
alpha = torch.from_numpy(
    np.array(
        np_alpha,
        dtype=np.float32,
    ),
).to(device)

beta = torch.from_numpy(
    np.array(
        np_beta,
        dtype=np.float32,
    ),
).to(device)

r = np.array(
    [0.9 for i in range(len(np_alpha))],
    dtype=np.float32,
)

w = np.array(
    [1e-2 for i in range(len(np_alpha))],
    dtype=np.float32,
)


actor = Actor(T,e,r,w)

policy_ration = []
for time in range(GENERATE_TIME):
    polic_prob = actor.calc_ration(
                feature=load_data.feature[time].clone(),
                edges=load_data.adj[time].clone()
                persona = persona
                )
    policy_ration.append(polic_prob)
