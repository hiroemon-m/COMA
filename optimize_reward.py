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
import time

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
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0005)

        return

    def optimize(self, t: int):
        impact = 0
        feat = self.feats[t].to_dense().to(device)

        edge = self.edges[t].to_dense().to(device) 
        self.optimizer.zero_grad()
        norm = feat.norm(dim=1)[:, None] + 1e-8
        feat_norm = feat.div(norm)
        dot_product = torch.matmul(feat_norm, torch.t(feat_norm)).to(device)
        sim = torch.mul(edge, dot_product)
        print(sim.size(),self.model.alpha.size())
        sim = torch.mul(sim, self.model.alpha)
        sim = torch.add(sim, 0.001)
        costs = torch.mul(edge, self.model.beta)
        costs = torch.add(costs, 0.001)
        reward = torch.sum(torch.sub(sim, costs))

        print(torch.sum(feat),torch.sum(edge))
        print("BEtr",reward.sum())




        #if t > 0:
        #    trend = (torch.sum(self.feats[t-1],dim=0)>0).repeat(500,1)
        #    trend = torch.where(trend>0,1,0)
            #trend = torch.sum(self.feats[t-1],dim=0).repeat(500,1)
        #    trend =  (trend - self.feats[t])/len(self.feats[0][0])
            #trend =  (trend - self.feats[t])
    

        #    impact = (trend*self.model.gamma.view(500,-1)).sum()

    
                

        if t > 0:
            
            new_feature = torch.matmul(self.edges[t].to_dense(),self.feats[t])
            #new_feature = torch.matmul(new_feature,torch.t(new_feature))
            old_feature = torch.matmul(self.edges[t-1].to_dense(), self.feats[t-1])
            #old_feature = torch.matmul(old_feature,torch.t(old_feature))
            #print("nosub",torch.sum((new_feature - old_feature)))
            #print("sub",torch.sum(torch.abs(new_feature - old_feature)))
            #print("scalesub",torch.sum(torch.abs(new_feature - old_feature)+1e-4)/len(self.feats[t][0]))
            print("tr",torch.sum(torch.matmul(self.model.gamma.view(1,-1),(torch.abs(new_feature - old_feature)+1e-4)/len(self.feats[t][0]))))
            print("rd",reward.sum())
            reward += torch.sum(
                #self.model.gamma/torch.sqrt(torch.sum(torch.abs(new_feature - old_feature)**2))
                torch.matmul(self.model.gamma.view(1,-1),(torch.abs(new_feature - old_feature)+1e-4)/len(self.feats[t][0]))
                #self.model.gamma/(torch.sqrt(torch.abs(new_feature - old_feature)+1e-4)**2)

            )
        #    print(torch.abs(new_feature - old_feature)+1e-4)
        
        print("rs",torch.sum(reward))
  
        loss = -reward.sum()
        loss.backward()
        print(self.model.alpha.grad[:5])
        print(self.model.beta.grad[:5])
        del loss
        self.optimizer.step()

       


    def export_param(self,data_type,data_name):
        #gamma/NIPS/
        with open("optimize/{}/{}/model.param.data.test.fast".format(data_type,data_name), "w") as f:
            max_alpha = 1.0
            max_beta = 1.0
            max_gamma = 1.0

            for i in range(self.size):
                f.write(
                    "{},{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                        self.model.gamma[i].item() / max_gamma,
                    )
                )


if __name__ == "__main__":
    start = time.perf_counter()
    data_name = "DBLP"

    data = init_real_data(data_name)
    data_size = len(data.adj[0])

    print(data_size)


    alpha = torch.ones(data_size,1)
    beta = torch.ones(data_size,1)
    gamma = torch.ones(data_size,1)
    model = Model(alpha, beta, gamma)

    


    optimizer = Optimizer(data.adj, data.feature, model, data_size)
    data_type = "complete"

    for t in range(5):
        
        optimizer.optimize(t)
        optimizer.export_param(data_type,data_name)
    end = time.perf_counter()
    print((end-start)/60)