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
    def __init__(self, edges, feature, temper) -> None:
        self.edges = edges
        self.feature = feature.to(device)
        self.temper = temper
        #self.alpha = alpha
        self.alpha = 1
        #self.beta = beta
        self.beta = 0.1
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

        self.feature_t = self.feature.t()
        dot_product = torch.mm(self.feature, self.feature_t)
        reward = next_mat.mul(dot_product).mul(self.alpha)
  
        costs = next_mat.mul(self.beta)
  
        reward = reward.sub(costs)
   
        return reward



    #隣接行列を返す
    def state(self):
        #neighbor_mat = torch.mul(self.edges, self.edges)
        neighbor_mat = torch.mul(self.edges, self.edges)

        return neighbor_mat, self.feature
