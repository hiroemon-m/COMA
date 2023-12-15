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
    def __init__(self, T, e, r, w, persona) -> None:
        super().__init__()
        self.T = nn.Parameter(
            T.clone().detach().requires_grad_(True)
        )
        
        self.e = nn.Parameter(
            e.clone().detach().requires_grad_(True)
        )
        self.r = nn.Parameter(
          r.view(-1,1).clone().detach().requires_grad_(True)
        )
        self.W = nn.Parameter(
           w.view(-1,1).clone().detach().requires_grad_(True)
        )
        self.persona = persona
        

    def calc_ration(self,attributes, edges,persona):
        norm = attributes.norm(dim=1)[:, None] + 1e-8
        attributes = attributes.div_(norm)
        #attributes_t = attributes.t()
        calc_policy = torch.empty(len(persona[0]),32)
        edges = (edges > 0).float().to(device)
        #print(len(persona[0]))
        for i in range(len(persona[0])):

            tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
            r = self.r[i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
            feat_prob = torch.tanh(feat)
            x = torch.mm(feat, feat.t())
            #print("x",x.size())
            x = x.div(self.T[i]).exp().mul(self.e[i])          
            #print("x",x)
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-4)
            #x = torch.nan_to_num(x,nan=0.0)
            x = torch.tanh(x)
            calc_policy[i] = torch.sum(x,dim=1)

         
        return calc_policy

  


    def forward(
        self,attributes, edges
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = attributes.norm(dim=1)[:, None] 
        norm = norm + 1e-8
        attributes = attributes.div(norm)

        edges = (edges > 0).float().to(device)
        #隣接ノードと自分の特徴量を集約する
        #print(edges.size()) 32x32
        #print(attributes.size())32x2411
        tmp_tensor = torch.mm(self.persona,self.W) * torch.matmul(edges, attributes)
        #tmp_tensor = torch.matmul(edges, attributes)
        
        #feat =  0.5*attributes + 0.5*tmp_tensor
        r = torch.mm(self.persona,self.r)

        r = r + 1e-8
        feat = r * attributes + tmp_tensor * (1 - r)
        #print("feat",feat)
        feat_prob = torch.tanh(feat)

        #x = x.div((np.linalg.norm(feat.detach().clone()))*(np.linalg.norm(feat.detach().clone().t())))

        # Compute similarity
        x = torch.mm(feat, feat.t())
        #print(x)
        x = x.div(torch.matmul(self.persona,self.T)).exp().mul(torch.matmul(self.persona,self.e))

        min_values = torch.min(x, dim=0).values
        # # 各列の最大値 (dim=0 は列方向)
        max_values = torch.max(x, dim=0).values
        # Min-Max スケーリング
        x = (x - min_values) / ((max_values - min_values) + 1e-4)+1e-4

        x = torch.tanh(x)

        #print(x)


        return x, feat, feat_prob

        # エッジの存在関数
        # x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # x = torch.relu(x)
        # print("prob", x)
        # return x, feat



    def predict(
        self,attributes, edges
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return self.forward(attributes, edges)
