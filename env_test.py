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
    def __init__(self, agent_num,edges, feature,alpha,beta,gamma,persona) -> None:

        self.agent_num = agent_num
        self.edges = edges
        #隣接行列をエッジリストへ変換
        self.edge_index = edges
        self.feature = feature
        self.feature_t = self.feature.t()
        self.alpha = alpha.clone().detach().requires_grad_(True)
        self.beta = beta.clone().detach().requires_grad_(True)
        self.gamma = gamma.clone().detach().requires_grad_(True)
        self.persona = persona.clone().detach().requires_grad_(True)



    def reset(self, edges, attributes,persona):
        self.persona = persona.detach().clone().requires_grad_(True)
        self.edges = edges
        self.feature = attributes
        self.feature_t = self.feature.t()

     

        return self.edges,self.feature
    #一つ進める

    def step(self,next_feature,next_action,time):
        reward = 0
        #norm = next_feature.norm(dim=1)[:, None] + 1e-8
        #next_feature = next_feature.div(norm)
        impact = 0


    
        persona_gamma = torch.mm(self.persona[time],self.gamma.view(self.persona[time].size()[1],1))
        new_feature = torch.matmul(next_action,next_feature)
        new_feature = torch.matmul(new_feature,torch.t(new_feature))
        old_feature = torch.matmul(self.edges, self.feature)
        old_feature = torch.matmul(old_feature,torch.t(old_feature))
        impact = torch.sum(torch.abs(new_feature - old_feature)+1e-4,dim=1)           
        reward = reward + impact.view(self.agent_num,1)*persona_gamma.view(self.agent_num,1)
        #reward= reward + impact
        self.edges = next_action
        self.feature = next_feature
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature_norm = self.feature.div(norm)
        self.feature_t = self.feature_norm.t()
        dot_product = torch.mm(self.feature_norm, self.feature_t).to(device)
        sim = torch.mul(self.edges,dot_product).sum(1)
        persona_alpha = torch.mm(self.persona[time],self.alpha.view(self.persona[time].size()[1],1))
        sim_alpha = sim.view(self.agent_num,1)*(persona_alpha.view(self.agent_num,1))
        sim_add = torch.add(sim_alpha,0.001)
        persona_beta = torch.mm(self.persona[time],self.beta.view(self.persona[time].size()[1],1))
        costs = self.edges.sum(1).view(self.agent_num,1)*persona_beta.view(self.agent_num,1)
        costs_add = torch.add(costs, 0.001)
        reward = reward + torch.sub(sim_add, costs_add)
        print("edge",self.edges.sum(1).size())
        print("reward",reward.size())
        

        return reward
    


    #隣接行列を返す
    def state(self):
        
        neigbor = torch.mul(self.edges,self.edges)
        return neigbor, self.feature
