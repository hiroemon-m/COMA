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
        """サンプルをGumbel(0, 1)から取る"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self,logits, tau):
        """Gumbel-Softmaxのサンプリング"""
        gumbel_noise = self.sample_gumbel(logits.shape)
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)

    def gumbel_softmax(self,logits,hard=False):
        """Gumbel-Softmaxサンプリング、ハードサンプルもサポート"""
        y = self.gumbel_softmax_sample(logits, self.temperature)

        if hard:
            # ハードサンプル：one-hotにするが、勾配はソフトサンプルに基づく
            shape = y.size()
            _, max_idx = y.max(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y).scatter_(-1, max_idx, 1.0)
            y = (y_hard - y).detach() + y
    
        return y

  
    def calc_ration(self,attributes, edges,persona):

        calc_policy = torch.empty(5,len(persona[0][0]),self.agent_num,self.agent_num)
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float()

        for i in range(len(persona[0][0])):
            for t in range(5):

                trend = (torch.sum(attributes,dim=0)>0).repeat(500,1)
                #trend = (torch.sum(attributes,dim=0)).repeat(500,1)
                #trend = torch.where(trend>0,1,0)
                

                feat_prob = torch.empty(len(attributes),len(attributes[0]),2)
                tmp_tensor = self.W[i] * torch.matmul(edges, attributes) + trend
                r = self.r[i]
                feat = r * attributes + tmp_tensor * (1-r)
                feat_tanh = torch.tanh(feat)
                feat_prob[:,:,0] = 10 - feat_tanh * 10
                feat_prob[:,:,1] = feat_tanh * 10
                feat= self.gumbel_softmax(feat_prob,hard=True)
                feat = feat[:,:,1]
                norm = feat.norm(dim=1)[:, None] + 1e-8
                feat = feat.div(norm)
                x = torch.mm(feat, feat.t())
                x = x.div(self.T[i]+1e-8)
                #x = torch.clamp(x, max=78)
                x = torch.exp(x)
                x = x*self.e[i]   

                # Min-Max スケーリング
                min_values = torch.min(x, dim=0).values
                max_values = torch.max(x, dim=0).values
                x = (x - min_values) / ((max_values - min_values) + 1e-8)

                x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
                calc_policy[t][i] = x

            
        return calc_policy

  


    def forward(self,attributes, edges,time) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"trainで呼び出す"""
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float()
        edges_prob = 0

            
        for i in range(len(self.persona[0][0])):
            trend = (torch.sum(attributes,dim=0)>0).repeat(500,1)
            #trend = (torch.sum(attributes,dim=0)).repeat(500,1)
            #trend = torch.where(trend>0,1,0)
        

            feat_prob = torch.empty(len(attributes),len(attributes[0]),2)     
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes) + trend
            r = self.r[i]
            feat = r * attributes + tmp_tensor * (1-r)
            feat_tanh = torch.tanh(feat)
            feat_prob[:,:,0] = 10 - feat_tanh*10
            feat_prob[:,:,1] = feat_tanh*10
            feat= self.gumbel_softmax(feat_prob,hard=True)
            feat = feat[:,:,1]
            norm = feat.norm(dim=1)[:, None] + 1e-8
            feat = feat.div(norm)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i]+1e-8)
            x = torch.exp(x)
            x = x*self.e[i]

            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)

            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            x = self.persona[time][:,i]*x 
            edges_prob = edges_prob + x
        edges_prob = edges_prob + 1e-3
        


        return edges_prob



    
    def predict(self,attributes, edges,time):

        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float()
        attr_ber = torch.empty(len(attributes),len(attributes[0]),2)
        edge_ber = torch.empty(len(edges),len(edges[0]),2)
        edges_prob = 0
        attr_prob = 0
            
        for i in range(len(self.persona[0][0])):
            trend = (torch.sum(attributes,dim=0)>0).repeat(500,1)
            #trend = (torch.sum(attributes,dim=0)).repeat(500,1)
            #trend = torch.where(trend>0,1,0)

            feat_prob = torch.empty(len(attributes),len(attributes[0]),2)
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes) + trend
            feat = self.r[i] * attributes + tmp_tensor * (1-self.r[i])        
            feat_tanh = torch.tanh(feat)
            #print("fear",torch.sum(feat_tanh>0),feat_tanh.size())
            #print("fear",feat_tanh[0,:12])
            #print("fear",feat_tanh[0,12:24])
            #print("fear",feat_tanh[0,24:])
            feat_prob[:,:,0] = 10 - (feat_tanh * 10)
            feat_prob[:,:,1] = (feat_tanh * 10)
            feat_action= self.gumbel_softmax(feat_prob,hard=True)
            #print("fear_prob12",feat_prob[0,:12])
            #print("fear_prob24",feat_prob[0,12:24])
            #print("fear_prob36",feat_prob[0,24:])
            feat_action = feat_action[:,:,1]
            #print("fear12",feat[0,:12])
            #print("fear24",feat[0,12:24])
            #print("fear36",feat[0,24:])
            #print("count",torch.sum(attributes),torch.sum(feat))
            norm = feat_action.norm(dim=1)[:, None] + 1e-8
            feat_action = feat_action.div(norm)
            x = torch.mm(feat_action, feat_action.t())
            x = x.div(self.T[i])
            x = torch.exp(x)
            x = x*self.e[i]

            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)

            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            #x = torch.tanh(x)
            x = self.persona[time][:,i]*x
            edges_prob = edges_prob + x
            attr_prob = attr_prob + self.persona[time][:,i].view(-1,1)*feat_tanh
            #edges_prob =  torch.clamp(edges_prob + x ,min=0,max=1)
        
        #属性の調整
        attr_prob = attr_prob*10
        attr_ber[:,:,0] = 10 - attr_prob
        attr_ber[:,:,1] = attr_prob
        attr_action= self.gumbel_softmax(attr_ber)[:,:,1]
        #print("be",torch.sum(edges_prob.bernoulli()))
        #print("edg",torch.sum(edges))
        edge_prob = edges_prob*10
        edge_ber[:,:,0] = 10 - edge_prob
        edge_ber[:,:,1] = edge_prob
        edge_action= self.gumbel_softmax(edge_ber)[:,:,1]



        return edges_prob,edge_action, attr_action
    

    def test(self,edges,attributes,time):
        
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        probability = 0
        attr_prob = 0
        attr_ber = torch.empty(len(attributes),len(attributes[0]),2)
        edge_ber = torch.empty(len(edges),len(edges[0]),2)
            
        for i in range(len(self.persona[0][0])):
            trend = (torch.sum(attributes,dim=0)>0).repeat(500,1)
            #trend = (torch.sum(attributes,dim=0)).repeat(500,1)
            #trend = torch.where(trend>0,1,0)
  

            feat_prob = torch.empty(len(attributes),len(attributes[0]),2)
    
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes) + trend
            #torch.matmul(1-edges, attributes)
            r = self.r[i]
            feat = r * attributes + tmp_tensor * (1-r)
            feat_tanh = torch.tanh(feat)
           
            feat_prob[:,:,0] = 10 - (feat_tanh*10)
            feat_prob[:,:,1] = feat_tanh*10
            feat= self.gumbel_softmax(feat_prob,hard=True)
            feat = feat[:,:,1]
            norm = feat.norm(dim=1)[:, None] + 1e-8
            feat = feat.div(norm)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i])
            x = torch.exp(x)
            x = x*self.e[i]
    
            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)

            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            x = self.persona[time][:,i]*x
            probability = probability + x
            attr_prob = attr_prob + self.persona[time][:,i].view(-1,1)*feat_tanh


        #属性の調整

        attr_ber[:,:,0] = 10 - attr_prob*10
        attr_ber[:,:,1] = attr_prob*10
        attr_action= self.gumbel_softmax(attr_ber)[:,:,1]

        edge_prob = probability*10
        edge_ber[:,:,0] = 10 - edge_prob
        edge_ber[:,:,1] = edge_prob
        edge_action= self.gumbel_softmax(edge_ber)[:,:,1]

        return edge_action,probability,attr_prob,attr_action




