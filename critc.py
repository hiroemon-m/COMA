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


class Critic(nn.Module):
    def __init__(self, T, e, r, w,persona) -> None:
        super().__init__()
        self.T = nn.Parameter(
            torch.tensor(T).float().to(device), requires_grad=True
        )
        self.e = nn.Parameter(
            torch.tensor(e).float().to(device), requires_grad=True
        )
        self.r = nn.Parameter(
            torch.tensor(r).float().view(-1, 1).to(device), requires_grad=True
        )
        self.W = nn.Parameter(
            torch.tensor(w).float().view(-1, 1).to(device), requires_grad=True
        )
        self.persona = persona
  


    def forward(
        self,attributes, edges
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = attributes.norm(dim=1)[:, None] 
        norm = norm + 1e-8
        attributes_normalized = attributes.div(norm)
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
            #x = torch.clamp(x, max=75)
            #x = torch.exp(x)
            #expm1
            #print(torch.max(x))
            x = x*self.e[i]

            # Min-Max スケーリング
            #min_values = torch.min(x, dim=0).values
            #max_values = torch.max(x, dim=0).values
            #x = (x - min_values) / ((max_values - min_values) + 1e-8)
    


        return x, feat




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
            
        for i in range(len(self.persona[0])):
    
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes_normalized)
            r = self.r[i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
                #print("feat",feat)
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i])+1e-4
            #x = torch.clamp(x, max=75)
            #print("nan",x[x=="Nan"].sum())
            #print(x)
            #x = torch.exp(x)
                #expm1
                #print(torch.max(x))
            x = x*self.e[i]

                # Min-Max スケーリング
            #min_values = torch.min(x, dim=0).values
            #max_values = torch.max(x, dim=0).values
            #x = (x - min_values) / ((max_values - min_values) + 1e-8)

            print(x)

            #print(probability)
    

        return x


