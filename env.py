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
    def __init__(self, agent_num, edges, feature, temper,alpha,beta,persona) -> None:
        self.agent_num = agent_num
        self.edges = edges
        self.feature = feature.to(device)
        self.temper = temper
        self.alpha = alpha
        self.beta = beta
        self.persona = persona

        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div_(norm)
        self.feature_t = self.feature.t()



    def reset(self, edges, attributes):
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
        # 特徴量の正規化vactor.pyでやってる
        #norm = self.feature.norm(dim=1)[:, None] + 1e-8
        #self.feature = self.feature.div(norm)
        print(self.persona.shape)
        print(self.alpha.shape)
        self.feature_t = self.feature.t()
        dot_product = torch.mm(self.feature, self.feature_t).to(device)
        sim = torch.mul(self.edges,dot_product).sum(1)
        persona_alpha = torch.mm(self.persona,self.alpha.view(self.persona.size()[1],1))
        sim = torch.dot(sim,persona_alpha.view(self.agent_num))
        sim = torch.add(sim,0.001)
        persona_beta = torch.mm(self.persona,self.beta.view(self.persona.size()[1],1))
        costs = torch.dot(self.edges.sum(1), persona_beta.view(self.agent_num))
        costs = torch.add(costs, 0.001)
        reward = torch.sub(sim, costs)
        print(reward)
        return reward



    #隣接行列を返す
    def state(self):
        #neighbor_mat = torch.mul(self.edges, self.edges)
        neighbor_mat = torch.mul(self.edges, self.edges)

        return neighbor_mat, self.feature
