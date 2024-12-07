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

        return self.edges,self.feature

    #一つ進める
    def step(self,next_feature,next_action,time):
        

        persona_num = self.persona[time].size()[1]
        #norm = next_feature.norm(dim=1)[:, None] + 1e-8
        #next_feature = next_feature.div(norm)
        #if time>0:
        #    persona_gamma = torch.mm(self.persona[time],self.gamma.view(self.persona[time].size()[1],1))
        #    trend = (torch.sum( self.feature,dim=0)>0).repeat(500,1)
        #    trend = torch.where(trend>0,1,0)
        #    trend = (trend - next_feature)/self.feature.size()[1]
        #    impact = (trend*persona_gamma.view(500,-1)).sum()

        print("r1")
        persona_gamma = torch.mm(self.persona[time],self.gamma.view(persona_num,1))
        print("na",next_action)
        print("nf",next_feature)
        new_feature = torch.sparse.mm(next_action,next_feature)
        print("nfna",new_feature)
        new_feature = torch.sparse.mm(new_feature,new_feature.t())
        print("2")

        old_feature = torch.sparse.mm(self.edges, self.feature)
        old_feature = torch.sparse.mm(old_feature,old_feature.t())
        print("r3")
        sub = torch.sub(old_feature,new_feature).abs()
        print(self.feature[0].size())
        print("r4")

        sub = sub/(self.feature[0].size()[0])

        #impact = torch.sparse.sum(features,dim=1)  
        #impact = impact.to_dense().view(self.agent_num,1)
        #persona_gamma = persona_gamma.view(self.agent_num,1)

        #print(sub.size())
        #print(persona_gamma.size())
        print("r5")
                
        #reward = torch.sparse.mm(sub.t(),persona_gamma)
        reward = sub.t().multiply(persona_gamma)

        self.edges = next_action
        self.feature = next_feature
        norm = (self.feature.to_dense().norm(dim=1)[:, None] + 1e-8)
        self.feature_norm = (self.feature.to_dense()/norm).to_sparse()
        self.feature_norm_t = self.feature_norm.t()
        print("r6")

        dot_product = torch.sparse.mm(self.feature_norm, self.feature_norm_t)
        sim = dot_product.multiply(dot_product)
        persona_alpha = torch.mm(self.persona[time],self.alpha.view(self.persona[time].size()[1],1))
        #sims = torch.sparse.mm(sim,persona_alpha).to_sparse()
        sims = sim.multiply(persona_alpha).to_sparse()
        print("r7")

        persona_beta = torch.mm(self.persona[time],self.beta.view(self.persona[time].size()[1],1))
        #costs = torch.sparse.mm(self.edges,persona_beta).to_sparse()
        costs = self.edges.multiply(persona_beta).to_sparse()
        print("size",sims.size())
        print("size",costs.size())
        print("size",reward.size())
        reward = reward + sims - costs
        

        

        return reward
    


    #隣接行列を返す
    def state(self):
        neighbar = self.edges@self.edges

        
        return neighbar, self.feature
