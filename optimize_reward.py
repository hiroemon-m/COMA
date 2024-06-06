# Standard Library
import random
from enum import IntEnum

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# First Party Library
import config
from init_real_data import init_real_data

device = config.select_device


class Interest(IntEnum):
    RED = 2
    BLUE = 1


class Model(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()

        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)
        return


class Optimizer:
    def __init__(self, edges, feats, model: Model, size: int):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        return

    def optimize(self, t: int):
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device)
        self.optimizer.zero_grad()
        dot_product = torch.matmul(feat, torch.t(feat)).to(device)
        sim = torch.mul(edge, dot_product)
        sim = torch.mul(sim, self.model.alpha)
        sim = torch.add(sim, 0.001)

        costs = torch.mul(edge, self.model.beta)
        costs = torch.add(costs, 0.001)
        if t>1:
            impact = torch.sum(torch.mm(self.edges[t],self.feats[t]) - torch.mm(self.edges[t],self.feats[t])**2)
        else:
            impact = torch.sum(torch.mm(self.edges[t],self.feats[t])**2)

        #reward = torch.sub(sim, costs)
        impacts = torch.mul(impact,self.model.gamma)
        reward = sim - costs + impacts
        loss = -reward.sum()

        loss.backward()
        print(loss)
        del loss
        self.optimizer.step()
        
        #グラフデータの可視化
       


    def export_param(self):
        with open("gamma/NIPS/model.param.data.fast", "w") as f:
            max_alpha = 1.0
            max_beta = 1.0

            for i in range(self.size):
                f.write(
                    "{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                    )
                )


if __name__ == "__main__":
    # data = attr_graph_dynamic_spmat_NIPS(T=10)
    # data = attr_graph_dynamic_spmat_DBLP(T=10)
    # data = TwitterData(T=10)
    # data = attr_graph_dynamic_spmat_twitter(T=10)
    data = init_real_data()
    data_size = len(data.adj[0])

    alpha = torch.from_numpy(
        np.array(
            [1.0 for i in range(data_size)],
            dtype=np.float32,
        ),
    ).to(device)

    beta = torch.from_numpy(
        np.array(
            [1.0 for i in range(data_size)],
            dtype=np.float32,
        ),
    ).to(device)

    gamma = torch.from_numpy(
        np.array(
            [1.0 for i in range(data_size)],
            dtype=np.float32,
        ),
    ).to(device)
    model = Model(alpha, beta, gamma)
    #i = 8
    #あるノードにi関する情報を取り除く
    #list[tensor]のキモい構造なので
    #for n in range(5):
        #print(data.adj[n][i].sum())
        #print(data.feature[n][i].sum())
        #data.adj[n][i,:] = 0
        #data.adj[n][:,i] = 0
        #data.feature[n][i][:] = 0
        #print(data.adj[n][i].sum())
        #print(data.feature[n][i].sum())

   


    #data.adj[4][i,:] = 0
    #data.adj[4][:,i] = 0
    #data.feature[4][i][:] = 0





    optimizer = Optimizer(data.adj, data.feature, model, data_size)
   
    for t in range(5):
        optimizer.optimize(t)

    optimizer.export_param()