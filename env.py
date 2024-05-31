# Standard Library
import math

# Third Party Library
import networkx as nx
import numpy as np
import scipy
import torch

# First Party Library
import config

device = config.select_device


class Env:
    def __init__(self, agent_num,edges, feature,alpha,beta,persona) -> None:
        self.agent_num = agent_num
        self.edges = edges
        #隣接行列をエッジリストへ変換
        self.edge_index = edges
        self.feature = feature.to(device)
        #単位行列作成
        #全部1の行列のほうがいいかも〜
        self.alpha = alpha.clone().detach().requires_grad_(True)
        self.beta = beta.clone().detach().requires_grad_(True)
        #self.alpha = alpha.requires_grad_(True)
        #self.beta = beta.requires_grad_(True)
        self.persona = persona.clone().detach().requires_grad_(True)
        #self.persona = persona.requires_grad_(True)





    def reset(self, edges, attributes,persona):
        self.persona = persona.detach().clone().requires_grad_(True)
        #self.persona = persona.requires_grad_(True)
        self.edges = edges
        self.feature = attributes
        self.feature = self.feature
    

        return self.edges,self.feature
    #一つ進める

    def step(self,next_feature,next_action,time):
        self.edges = next_action
        self.feature = next_feature
        self.feature_t = self.feature.t()
        dot_product = torch.mm(self.feature, self.feature_t).to(device)
        sim = torch.mul(self.edges,dot_product).sum(1)
        persona_alpha = torch.mm(self.persona[time],self.alpha.view(self.persona[time].size()[1],1))
        sim_alpha = sim.view(self.agent_num,1)*(persona_alpha.view(self.agent_num,1))
        sim_add = torch.add(sim_alpha,0.001)
        persona_beta = torch.mm(self.persona[time],self.beta.view(self.persona[time].size()[1],1))
        costs = self.edges.sum(1).view(self.agent_num,1)*persona_beta.view(self.agent_num,1)
        costs_add = torch.add(costs, 0.001)
        reward = torch.sub(sim_add, costs_add)
        self.sim = sim.sum()
        self.costs = costs.sum()
        self.persona_alpha = persona_alpha
        self.persona_beta = persona_beta

        return reward
    


    #隣接行列を返す
    def state(self):
        #neighbor_mat = torch.mul(self.edges, self.edges)
        #neighbor_mat = torch.mul(self.edges, self.edges)

        return self.edges, self.feature
