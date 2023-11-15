# Standard Library
import math

# Third Party Library
import networkx as nx
import numpy as np
import scipy
import torch.nn as nn
import torch

# First Party Library
import config

device = config.select_device


class Env:
    def __init__(self, agent_num, edges, feature,alpha,beta,persona) -> None:
        self.agent_num = agent_num
        self.edges = edges
        self.feature = feature.to(device)
        #self.alpha = alpha
        #self.beta = beta
        self.alpha = alpha.clone().detach().requires_grad_(True)
        
        self.beta = beta.clone().detach().requires_grad_(True)
        self.persona = persona.clone().detach().requires_grad_(True)

        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div_(norm)
        self.feature_t = self.feature.t()
        self.sim = 0
        self.costs = 0
        self.persona_alpha = 0
        self.persona_beta = 0



    def reset(self, edges, attributes,alpha,beta,persona):

        self.alpha = alpha.detach().clone().requires_grad_(True)
        #self.alpha = alpha.requires_grad_(True)
        self.beta = beta.detach().clone().requires_grad_(True)
        #self.beta = beta.requires_grad_(True)
        self.persona = persona.detach().clone().requires_grad_(True)
        #self.persona = persona.requires_grad_(True)
        self.edges = edges
        self.feature = attributes
        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div(norm)
        self.feature_t = self.feature.t()

        return self.edges,self.feature
    #一つ進める

    def step(self,feature,action):
        next_mat = action
        self.edges = next_mat
        next_feature = feature
        self.feature = next_feature
        self.feature_t = self.feature.t()
        dot_product = torch.mm(self.feature, self.feature_t).to(device)
        sim = torch.mul(self.edges,dot_product).sum(1)
        persona_alpha = torch.mm(self.persona,self.alpha.view(self.persona.size()[1],1))
        sim_alpha = sim.view(self.agent_num,1)*(persona_alpha.view(self.agent_num,1))
        sim_add = torch.add(sim_alpha,0.001)
        persona_beta = torch.mm(self.persona,self.beta.view(self.persona.size()[1],1))
        costs = self.edges.sum(1).view(self.agent_num,1)*persona_beta.view(self.agent_num,1)
        costs_add = torch.add(costs, 0.001)
        reward = torch.sub(sim_add, costs_add)
        self.sim = sim.sum()
        self.costs = costs.sum()
        self.persona_alpha = persona_alpha
        self.persona_beta = persona_beta
        #reward.sum().backward()
        #print("persona",self.persona.is_leaf,self.persona.grad,self.persona.grad_fn)
        #print("alpha",self.alpha.is_leaf,self.alpha.grad)
        #print("pealpha",persona_alpha.grad_fn)
        #print("beta",self.beta.grad)
        #print("pebeta",persona_beta.grad_fn)
        #print("sim",sim.grad,sim.grad_fn)
        #print("costs",costs.grad,costs.grad_fn)
        return reward



    #隣接行列を返す
    def state(self):
        #neighbor_mat = torch.mul(self.edges, self.edges)
        neighbor_mat = torch.mul(self.edges, self.edges)
        #print("mat",neighbor_mat.shape)
        #
        # print(self.feature.shape)

        return neighbor_mat, self.feature
