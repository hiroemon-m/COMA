# Standard Library
from typing import Tuple

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# First Party Library
import math
import config
import csv

device = config.select_device


class Actor(nn.Module):

    def __init__(self, T, e, r, w,persona,agent_num,temperature) -> None:

        super().__init__()
        self.T = nn.Parameter(T.clone().detach().requires_grad_(True))
        self.e = nn.Parameter(e.clone().detach().requires_grad_(True))
        self.r = nn.Parameter(r.clone().detach().to(device).requires_grad_(True))
        self.W = nn.Parameter(w.clone().detach().to(device).requires_grad_(True))
        self.temperature = temperature
        self.persona = persona
        self.agent_num = agent_num
        

    def sample_gumbel(self,shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self,logits, temperature):
        gumbel_noise = self.sample_gumbel(logits.size())
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self,logits):
        y = self.gumbel_softmax_sample(logits, self.temperature)
        return y

    def calc_ration(self,attributes, edges,persona):

        calc_policy = torch.empty(5,len(persona[0][0]),self.agent_num,self.agent_num)
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)

        for i in range(len(persona[0][0])):
            for t in range(5):
                tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
                r = self.r[i]

                feat = r * attributes + tmp_tensor * (1-r)
                # Min-Max スケーリング
                #min_values = torch.min(feat, dim=0).values
                #max_values = torch.max(feat, dim=0).values
                #x = (feat - min_values) / ((max_values - min_values) + 1e-8)
                #x = torch.mm(x, x.t())
                feat = torch.tanh(feat)
                logits = torch.log((feat/(1-feat+1e-4))+math.e)
                feat= self.gumbel_softmax(logits)

                x = torch.mm(feat, feat.t())
                x = x.div(self.T[i]+1e-8)
                #x = torch.clamp(x, max=78)
                x = torch.exp(x)
                x = x*self.e[i]          
                x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
                x = torch.tanh(x)
                calc_policy[t][i] = x

            
        return calc_policy

  


    def forward(self,attributes, edges,time) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"trainで呼び出す"""
        attr= 0
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        edges_prob = 0

        for i in range(len(self.persona[0][0])):
            
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
            r = self.r[i]
    
            feat = r * attributes + tmp_tensor * (1-r)
            # Min-Max スケーリング
            #min_values = torch.min(feat, dim=0).values
            #max_values = torch.max(feat, dim=0).values
            #x = (feat - min_values) / ((max_values - min_values) + 1e-8)
            #x = torch.mm(x, x.t())
            feat_prob = torch.tanh(feat)
            logits = torch.log((feat_prob/(1-feat_prob+1e-4))+math.e)
            feat= self.gumbel_softmax(logits)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i]+1e-8)
            #x = torch.clamp(x,max=78)
            x = torch.exp(x)
            x = x*self.e[i]

            # Min-Max スケーリング
            #min_values = torch.min(x, dim=0).values
            #max_values = torch.max(x, dim=0).values
            #x = (x - min_values) / ((max_values - min_values) + 1e-8)
            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            x = self.persona[time][:,i]*x
            edges_prob = edges_prob + x
            #edges_prob =  torch.clamp(edges_prob + x ,min=0,max=1)
            #edges_prob = edges_prob + 1e-10
            attr = attr + self.persona[time][:,i].view(-1,1)*feat_prob

        #属性の方策
        attr_action = torch.where(attr > 0.5, torch.tensor(1.0), torch.tensor(0.0))
    

        return edges_prob, attr_action




    def predict(self,attributes, edges,time):

        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
       
        edges_prob = 0
        attr_prob = 0
            
        for i in range(len(self.persona[0][0])):
            
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
            r = self.r[i]
            feat = r * attributes + tmp_tensor * (1-r)
            # Min-Max スケーリング
            #min_values = torch.min(feat, dim=0).values
            #max_values = torch.max(feat, dim=0).values
            #x = (feat - min_values) / ((max_values - min_values) + 1e-8)
            #x = torch.mm(x, x.t())

            feat_prob = torch.tanh(feat)
            logits = torch.log((feat_prob/(1-feat_prob+1e-4))+math.e)
            feat= self.gumbel_softmax(logits)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i]+1e-8)
            #x = torch.clamp(x, max=78)
            x = torch.exp(x)
            x = x*self.e[i]

            #min_values = torch.min(x, dim=0).values
            #max_values = torch.max(x, dim=0).values
            #x = (x - min_values) / ((max_values - min_values) + 1e-8)
            
            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            x = self.persona[time][:,i]*x
            edges_prob = edges_prob + x
            attr_prob = attr_prob + self.persona[time][:,i].view(-1,1)*feat_prob
            #edges_prob =  torch.clamp(edges_prob + x ,min=0,max=1)

        #属性の調整
        attr_action = torch.where(attr_prob > 0.5, torch.tensor(1.0), torch.tensor(0.0))


        return edges_prob, attr_action
    

    def test(self,edges,attributes,time):
        
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        probability = 0
        attr_prob = 0
            
        for i in range(len(self.persona[0][0])):
    
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
            r = self.r[i]
            feat = r * attributes + tmp_tensor * (1-r)
            feat_prob = torch.tanh(feat)
            logits = torch.log((feat_prob/(1-feat_prob+1e-4))+math.e)
            feat= self.gumbel_softmax(logits)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i]+1e-8)
            #x = torch.clamp(x, max=78)
            x = torch.exp(x)
            x = x*self.e[i]
            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            x = self.persona[time][:,i]*x
            #probability =  torch.clamp(probability + x ,min=0,max=1)
            probability = probability + x

            attr_prob = attr_prob + self.persona[time][:,i].view(-1,1)*feat_prob

        #属性の調整
        attr_action = torch.where(attr_prob > 0.5, torch.tensor(1.0), torch.tensor(0.0))
   

        return probability,attr_prob,attr_action




