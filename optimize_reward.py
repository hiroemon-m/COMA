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
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)

        return

    def optimize(self, time: int):

        for t in range(time):
            feat = self.feats[t].to(device)
            edge = self.edges[t].to(device) 
            self.optimizer.zero_grad()
            norm = feat.norm(dim=1)[:, None] + 1e-8
            feat_norm = feat.div(norm)
            dot_product = torch.matmul(feat_norm, torch.t(feat_norm)).to(device)
            sim = torch.mul(edge, dot_product)
            sim = torch.mul(sim, self.model.alpha[t])
            sim = torch.add(sim, 0.001)
            costs = torch.mul(edge, self.model.beta[t])
            costs = torch.add(costs, 0.001)
            reward = torch.sub(sim, costs)

            if t > 0:
                print(self.edges[t])
                new_feature = torch.matmul(self.edges[t],self.feats[t])
                old_feature = torch.matmul(self.edges[t-1], self.feats[t-1])
            

                reward += torch.sum((
                    torch.matmul(self.model.gamma[t].view(1,-1),(torch.abs(new_feature - old_feature)+1e-4))) 
                )


    
            loss = -reward.sum()
            loss.backward()
            del loss
            self.optimizer.step()

       


    def export_param(self):
        #gamma/NIPS/
        for t in range(time):
            with open("gamma/NIPS/model_param_time={}".format(t), "w") as f:
                max_alpha = 1.0
                max_beta = 1.0
                max_gamma = 1.0
                for i in range(self.size):
                    f.write(
                        "{},{},{}\n".format(
                            self.model.alpha[t][i].item() / max_alpha,
                            self.model.beta[t][i].item() / max_beta,
                            self.model.gamma[t][i].item() / max_gamma,
                        )
                        )


if __name__ == "__main__":
    # data = attr_graph_dynamic_spmat_NIPS(T=10)
    # data = attr_graph_dynamic_spmat_DBLP(T=10)
    # data = TwitterData(T=10)
    # data = attr_graph_dynamic_spmat_twitter(T=10)
    time = 5
    data = init_real_data()
    data_size = len(data.adj[0])

    alpha = torch.from_numpy(
        np.array(
            [[1.0 for i in range(data_size)] for t in range(time)],
            dtype=np.float32,
        ),
    ).to(device)

    beta = torch.from_numpy(
        np.array(
            [[1.0 for i in range(data_size)] for t in range(time)],
            dtype=np.float32,
        ),
    ).to(device)

    gamma = torch.from_numpy(
        np.array(
            [[1.0 for i in range(data_size)] for t in range(time)],
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

    
    optimizer.optimize(time)
    optimizer.export_param()