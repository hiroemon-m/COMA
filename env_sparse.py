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
    def __init__(self, agent_num,edge, feature,alpha,beta,gamma,persona) -> None:

        self.agent_num = agent_num
        self.edges = edge
        #隣接行列をエッジリストへ変換
        self.feature = feature
        self.feature_t = self.feature.t()
        self.alpha = alpha.clone().detach().requires_grad_(True)
        self.beta = beta.clone().detach().requires_grad_(True)
        self.gamma = gamma.clone().detach().requires_grad_(True)
        self.persona = persona.clone().detach().requires_grad_(True)
        #norm = self.feature.norm(dim=1)[:, None] + 1e-8
        #self.feature = self.feature.div(norm)
        


    def reset(self, edges, attributes,persona):
        self.persona = persona.detach().clone().requires_grad_(True)
        self.edges = edges
        self.feature = attributes
        self.feature_t = self.feature.t()
       
        #norm = attributes.norm(dim=1)[:, None] + 1e-8
        #self.feature = attributes.div(norm)
     

        return self.edges,self.feature
    #一つ進める

    def step(self,next_feature,next_action,time):
        #agent_num = self.persona[time].size()[0]
        persona_num = self.persona[time].size()[1]
        #norm = next_feature.norm(dim=1)[:, None] + 1e-8
        #next_feature = next_feature.div(norm)
        #if time>0:
        #    persona_gamma = torch.mm(self.persona[time],self.gamma.view(self.persona[time].size()[1],1))
        #    trend = (torch.sum( self.feature,dim=0)>0).repeat(500,1)
        #    trend = torch.where(trend>0,1,0)
        #    trend = (trend - next_feature)/self.feature.size()[1]
        #    impact = (trend*persona_gamma.view(500,-1)).sum()

    
        persona_gamma = torch.mm(self.persona[time],self.gamma.view(persona_num,1))
        new_feature = torch.sparse.mm(next_action,next_feature)
        new_feature = torch.sparse.mm(new_feature,torch.t(new_feature))
        old_feature = torch.sparse.mm(self.edges, self.feature)
        old_feature = torch.sparse.mm(old_feature,torch.t(old_feature))
        diff_feature = new_feature -old_feature
        abs_values = torch.abs(diff_feature.values())
        features = torch.sparse_coo_tensor(diff_feature.indices(),abs_values,diff_feature.size())
        #impact = torch.sparse.sum(features,dim=1)  
        #impact = impact.to_dense().view(self.agent_num,1)
        #persona_gamma = persona_gamma.view(self.agent_num,1)
        #print(features.size())
        #print(persona_gamma.size())
        reward = features*persona_gamma

        #reward= reward + impact
        self.edges = next_action
        self.feature = next_feature
        norm = (self.feature.to_dense().norm(dim=1)[:, None] + 1e-8)
        self.feature_norm = (self.feature.to_dense()/norm).to_sparse()
        self.feature_norm_t = self.feature_norm.t()
        dot_product = torch.sparse.mm(self.feature_norm, self.feature_norm_t)
        sim = torch.sparse.mm(self.edges,dot_product)
        #sim = torch.sparse.sum(sim,dim=1)
        persona_alpha = torch.mm(self.persona[time],self.alpha.view(self.persona[time].size()[1],1))
        #sim = sim.to_dense().view(self.agent_num,1)
        #sims = sim.to_sparse()*(persona_alpha.view(self.agent_num,1))
        sims = sim*persona_alpha
        persona_beta = torch.mm(self.persona[time],self.beta.view(self.persona[time].size()[1],1))
        #edge_sum = torch.sparse.sum(self.edges,dim=1)
        #edge_sum = edge_sum.to_dense().view(self.agent_num,1)
        #costs = edge_sum.to_sparse()*persona_beta.view(self.agent_num,1)
        costs = self.edges*persona_beta
        reward = reward + sims - costs

        

        return reward
    


    #隣接行列を返す
    def state(self):
        
        return self.edges, self.feature
