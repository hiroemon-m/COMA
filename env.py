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
        self.edges = edges.repeat(len(persona[0][0]),1,1)
        #隣接行列をエッジリストへ変換
        self.edge_index = edges
        self.feature = feature.repeat(len(persona[0][0]),1,1)
      
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
        print("feat",torch.sum(next_feature,dim=-1))
        print("edge",torch.sum(next_action,dim=-1))
        reward = torch.empty([len(self.persona[0][0]),self.agent_num,self.agent_num])
        impact = 0

        persona_t = torch.transpose(self.persona[time],1,0)
        persona_alpha = torch.unsqueeze(persona_t*self.alpha,dim=2).expand(len(self.persona[0][0]),self.agent_num,self.agent_num)
        persona_beta = torch.unsqueeze(persona_t*self.beta,dim=2).expand(len(self.persona[0][0]),self.agent_num,self.agent_num)
        persona_gamma = torch.unsqueeze(persona_t*self.gamma,dim=2).expand(len(self.persona[0][0]),self.agent_num,self.agent_num)

        # (persona_num,agent_num,agent_num)→agent_num
        new_feature = torch.matmul(next_action,next_feature)
        old_feature = torch.matmul(self.edges, self.feature)
        impact = torch.sum(torch.abs(new_feature - old_feature)+1e-4,dim=2) 
        impact_matrix = torch.unsqueeze(impact,dim=1).expand(len(self.persona[0][0]),self.agent_num,self.agent_num)#32,32
        impact_reward = persona_gamma.expand(len(self.persona[0][0]),self.agent_num,self.agent_num)*impact_matrix#32,32
        self.edges = next_action
        self.feature = next_feature
        #全体でnorm ?perusonaごとは微妙？
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature_norm = self.feature.div(norm)
        self.feature_t = torch.transpose(self.feature_norm,2,1)
        #検討
        dot_product = torch.matmul(self.feature_norm, self.feature_t).to(device)#32,32
        sim = self.edges*dot_product#5,32,32
        sim_reward = sim*persona_alpha

        cost_reward = persona_beta*self.edges
        reward = torch.sub(sim_reward, cost_reward) + impact_reward

        reward_clip=[-1,1]
        reward = torch.clamp(reward,min=reward_clip[0],max=reward_clip[1])
        print("rewaerd",reward.sum())
        return reward
    
    def test_step(self,next_feature,next_action,time):
        reward = 0
        #norm = next_feature.norm(dim=1)[:, None] + 1e-8
        #next_feature = next_feature.div(norm)
        impact = 0
        #if time>0:
        #    persona_gamma = torch.mm(self.persona[time],self.gamma.view(self.persona[time].size()[1],1))
        #    trend = (torch.sum( self.feature,dim=0)>0).repeat(500,1)
        #    trend = torch.where(trend>0,1,0)
        #    trend = (trend - next_feature)/self.feature.size()[1]
        #    impact = (trend*persona_gamma.view(500,-1)).sum()

    
        persona_gamma = torch.mm(self.persona[time],self.gamma.view(self.persona[time].size()[1],1))
        new_feature = torch.matmul(next_action,next_feature)
        old_feature = torch.matmul(self.edges, self.feature)
       
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

        

        return reward
    


    


    #隣接行列を返す
    def state(self):

        return self.edges, self.feature
