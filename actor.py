# Standard Library
from typing import Tuple

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# First Party Library
import config
import csv
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
device = config.select_device


class Actor(nn.Module):
    def __init__(self, T, e, r, w,persona) -> None:
        super().__init__()
        self.T = nn.Parameter(T.clone().detach().requires_grad_(True))
        self.e = nn.Parameter(e.clone().detach().requires_grad_(True))
        self.r = nn.Parameter(r.clone().detach().to(device).requires_grad_(True))
        self.W = nn.Parameter(w.clone().detach().to(device).requires_grad_(True))

        self.persona = persona

    def calc_ration(self,attributes, edges,persona):

        calc_policy = torch.empty(len(persona[0]),32,32)
        norm = attributes.norm(dim=1)[:, None] 
        norm = norm + 1e-8
        attributes = attributes/norm
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)

        for i in range(len(persona[0])):
        
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
            r = self.r[i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i])+1e-8
            x = torch.clamp(x, max=78)
            x = torch.exp(x)
            x = x*self.e[i]

            #x = torch.clamp(x, max=75)
            #x=x.exp().mul(self.e[i])
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = ((x - min_values)) / ((max_values - min_values) + 1e-8)
        
            
            x = torch.tanh(x)
            #calc_policy[i] = torch.sum(x,dim=1)
            calc_policy[i] = x

            #print("xik",x.size())

         
        return calc_policy

  


    def forward(
        self,attributes, edges
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = attributes.norm(dim=1)[:, None] 
        norm = norm + 1e-8
        attributes_normalized = attributes/norm
        #print(self.persona[:,1])
        #print(self.persona)
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        #edges = (edges > 0).float().to(device)

        probability = 0
        
        for i in range(len(self.persona[0])):
 
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes_normalized)
            r = self.r[i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
            #print("feat",feat)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i])+1e-8
            x = torch.clamp(x, max=78)
            x = torch.exp(x)

            x = x*self.e[i]

            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)
     
            x = torch.tanh(x)
            x = self.persona[:,i]*x
    
            probability =  torch.clamp(probability + x ,min=0,max=1)
            #属性の調整
            feat = torch.where(feat>0,1,feat)
            feat = torch.where(feat>1,1,feat)

        return probability, feat




    def predict(self,attributes, edges):
        
        norm = attributes.norm(dim=1)[:, None] 
        norm = norm + 1e-8
        attributes_normalized = attributes.div(norm)

            #print(self.persona[:,1])
            #print(self.persona)
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        #edges = (edges > 0).float().to(device)
        probability = 0
        probability_tensor = torch.empty(len(self.persona[1]),len(self.persona[1]),requires_grad=True)
            
        for i in range(len(self.persona[0])):
    
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes_normalized)
            r = self.r[i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
                #print("feat",feat)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i])+1e-8
            #x = torch.clamp(x, max=75)
            #print("nan",x[x=="Nan"].sum())
            #print(x)
            #x = torch.exp(x)
                #expm1
                #print(torch.max(x))
            x = torch.clamp(x, max=78)
            x = torch.exp(x)
            x = x*self.e[i]
  


                # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)
            x = torch.tanh(x)
            x = self.persona[:,i]*x

            #print(y)
            probability =  torch.clamp(probability + x ,min=0,max=1)
        #probability_tensor =  probability.clone()
        #print("pr",probability)
        #print(probability_tensor)
                        #属性の調整
            feat = torch.where(feat>0,1,feat)
            feat = torch.where(feat>1,1,feat)

    

        return probability, feat


