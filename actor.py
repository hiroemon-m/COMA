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

  
    def calc_ration(self,attributes, edges,persona,past_feature):

        calc_policy = torch.empty(5,len(persona[0][0]),self.agent_num,self.agent_num)
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float()

        for i in range(len(persona[0][0])):
            past_feature_t = past_feature
            for t in range(5):

                trend = (torch.sum(attributes,dim=0)>0).repeat(500,1)
                #trend = (torch.sum(attributes,dim=0)).repeat(500,1)
                #trend = torch.where(trend>0,1,0)
                feat_prob = torch.empty(len(attributes),len(attributes[0]),2)
                #tmp_tensor = self.W[i] * torch.matmul(edges, attributes) + trend

                tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
                r = self.r[i]
                feat = r * attributes + tmp_tensor * (1-r)
                #feat = r * past_feature_t + tmp_tensor * (1-r)
                feat_tanh = torch.tanh(feat)
                feat_prob[:,:,0] = 10 - feat_tanh * 10
                feat_prob[:,:,1] = feat_tanh * 10
                feat= self.gumbel_softmax(feat_prob,hard=True)
                feat = feat[:,:,1]
                past_feature_t = 0.8*past_feature_t + feat
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

  


    def forward(self,attributes, edges,time,past_feature) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"trainで呼び出す"""
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float()
        edge_prob = torch.empty(len(self.persona[0][0]),len(edges[0]),len(edges[0][0])) 
       

        feat_prob = torch.empty(len(self.persona[0][0]),len(attributes[0]),len(attributes[0][0]),2)
        #tmp_tensor = self.W[i] * torch.matmul(edges, attributes) + trend
        tmp_tensor = self.W * torch.matmul(edges, attributes)
        r = self.r

        feat = r * attributes + tmp_tensor * (1-r)
        #feat = r * past_feature + tmp_tensor * (1-r)


        feat_tanh = torch.tanh(feat)
        feat_prob[:,:,:,0] = 10 - (feat_tanh * 10)
        feat_prob[:,:,:,1] = (feat_tanh * 10)
        feat_action= self.gumbel_softmax(feat_prob,hard=True)
        feat_action = feat_action[:,:,:,1]

        norm = feat_action.norm(dim=1)[:, None] + 1e-8
        feat_action = feat_action.div(norm)
        feat_t = torch.transpose(feat_action,2,1)
        x = torch.matmul(feat_action, feat_t)
        x = x.div(self.T)
        x = torch.clamp(x,max=87)

        x = torch.exp(x)
        x = x*self.e

        # Min-Max スケーリング
        min_values = torch.min(x, dim=2).values
        max_values = torch.max(x, dim=2).values
        x = (x - min_values.unsqueeze(-1)) / ((max_values - min_values).unsqueeze(-1) + 1e-8)
        x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
        edge_prob  = x +1e-5

        past_feature = 0.8*past_feature 


        return edge_prob,past_feature



    
    def predict(self,attributes, edges,time,past_feature):

        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float()
        attr_ber = torch.empty(len(self.persona[0][0]),len(attributes[0]),len(attributes[0][0]),2)
        edge_ber = torch.empty(len(self.persona[0][0]),len(edges[0]),len(edges[0]),2)
        edge_prob = torch.empty(len(self.persona[0][0]),len(edges[0]),len(edges[0][0])) 
        attr_prob = torch.empty(len(self.persona[0][0]),len(attributes[0]),len(attributes[0][0]))
        feat_prob = torch.empty(len(self.persona[0][0]),len(attributes[0]),len(attributes[0][0]),2)
        #r,w:4,1,1

        tmp_tensor = self.W * torch.matmul(edges, attributes)
        r = self.r
        #4,32,3000
        feat = r * attributes + tmp_tensor * (1-r)
        #feat = r * past_feature + tmp_tensor * (1-r)


        feat_tanh = torch.tanh(feat)
        feat_action = feat
        feat_prob[:,:,:,0] = 10 - (feat_tanh * 10)  
        feat_prob[:,:,:,1] = (feat_tanh * 10)
        feat_action= self.gumbel_softmax(feat_prob,hard=True)
        del feat_prob
        feat_action = feat_action[:,:,:,1]
        norm = feat_action.norm(dim=1)[:, None] + 1e-8
        feat_action = feat_action.div(norm)
        feat_t = torch.transpose(feat_action,2, 1)
        x = torch.matmul(feat_action, feat_t)
        x = x.div(self.T)
        x = torch.clamp(x,max=87)
    
        x = torch.exp(x)
        x = x*self.e

        # Min-Max スケーリング
        min_values = torch.min(x, dim=2).values
        max_values = torch.max(x, dim=2).values
        x = (x - min_values.unsqueeze(-1)) / ((max_values - min_values).unsqueeze(-1) + 1e-8)
        x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))

        edge_prob  = x +1e-5
        attr_prob = feat_tanh

        #属性の調整
        attr_ber[:,:,:,0] = 10 - attr_prob*10
        attr_ber[:,:,:,1] = attr_prob*10
        attr_action= self.gumbel_softmax(attr_ber)[:,:,:,1]
        #エッジの調整
        edge_ber[:,:,:,0] = 10 - edge_prob*10
        edge_ber[:,:,:,1] = edge_prob*10
        edge_action= self.gumbel_softmax(edge_ber)[:,:,:,1]

        past_feature = 0.8*past_feature + attr_action

        return edge_prob,edge_action,attr_prob,attr_action,past_feature
    

    def test(self,edges,attributes,time,past_feature):
        
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)

        probability = 0
        attr_prob = 0
        attr_ber = torch.empty(len(attributes),len(attributes[0]),2)
        edge_ber = torch.empty(len(edges),len(edges[0]),2)

            
        for i in range(len(self.persona[0][0])):
            
            feat_prob = torch.empty(len(attributes),len(attributes[0]),2)
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes) 
            r = self.r[i]
            feat = r * attributes + tmp_tensor * (1-r)
            #feat = r * past_feature + tmp_tensor * (1-r)

            feat_tanh = torch.tanh(feat)
            feat_prob[:,:,0] = 10 - (feat_tanh*10)
            feat_prob[:,:,1] = feat_tanh*10
            feat_action= self.gumbel_softmax(feat_prob,hard=True)
            feat_action = feat_action[:,:,1]
            norm = feat_action.norm(dim=1)[:, None] + 1e-8
            feat_norm = feat.div(norm)
            x = torch.mm(feat_norm, feat_norm.t())
            x = x.div(self.T[i])
            x = torch.exp(x)
            x = x*self.e[i]
    
            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)
            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))

            probability = probability + self.persona[time][:,i]*x
            attr_prob = attr_prob + self.persona[time][:,i].view(-1,1)*feat_tanh


        #属性の調整

        attr_ber[:,:,0] = 10 - attr_prob*10
        attr_ber[:,:,1] = attr_prob*10
        attr_action= self.gumbel_softmax(attr_ber)[:,:,1]

        edge_prob = probability*10
        edge_ber[:,:,0] = 10 - edge_prob
        edge_ber[:,:,1] = edge_prob
        edge_action= self.gumbel_softmax(edge_ber)[:,:,1]

        past_feature = 0.8*past_feature + attr_action

        return edge_action,probability,attr_prob,attr_action,past_feature




