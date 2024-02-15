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

    def calc_ration(self,attributes, edges,persona):

        calc_policy = torch.empty(len(persona[0]),32)
        edges = (edges > 0).float().to(device)
        for i in range(len(persona[0])):
        
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
            r = self.r[i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
            feat_prob = feat
            x = torch.mm(feat, feat.t())
            x = x.div(self.T[i])
            x = torch.clamp(x, max=75)
            #x = torch.clamp(x, max=73)
            #x = torch.expm1(x).mul(self.e[i])
            x=x.exp().mul(self.e[i])
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-4)
        
            
            x = torch.tanh(x)
            #print("before",x[0])
            #print(persona[:,i])
            #persona_matrix = persona[:,i].view(-1, 1).repeat(1, 32)
            #print(persona_matrix[0])
            #print(persona_matrix[:,0])
            #x =  persona_matrix*x
            #x = torch.matmul(x,persona[:,i])
            #print("after",x[0])
        
            #m = nn.ReLU(inplace=False)
            #x = m(x)
            calc_policy[i] = torch.sum(x,dim=1)

         
        return calc_policy

  


    def forward(
        self,attributes, edges
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = attributes.norm(dim=1)[:, None] 
        norm = norm + 1e-8
        attributes = attributes.div(norm)

        #print(self.persona[:,1])
        #print(self.persona)
        edges = (edges > 0).float().to(device)
        probability = 0
        next_feat = 0
        for i in range(len(self.persona[0])):
 
            #隣接ノードと自分の特徴量を集約する
            #print(edges.size()) 32x32
            #print(attributes.size())32x2411
            tmp_tensor = (self.W[i]+1e-4) * torch.matmul(edges, attributes)
            #tmp_tensor = torch.matmul(edges, attributes)
       
      
         
            #feat =  0.5*attributes + 0.5*tmp_tensor
            r = self.r[i]

            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
          

            next_feat += feat

            #print("feat",feat)
            #feat_prob = torch.sigmoid(feat)
            feat_prob = feat

            #x = x.div((np.linalg.norm(feat.detach().clone()))*(np.linalg.norm(feat.detach().clone().t())))

            # Compute similarity
            x = torch.mm(feat, feat.t())
            #print(attributes,tmp_tensor)
            #print("terw",self.r,self.W)
            #print("max",torch.max(x))
            #print(self.T)
            #print(x)
            x = x.div(self.T[i])+1e-4
            #print("max_x",torch.max(x))
            #print("x",torch.isnan(x).sum())
           
            #x = torch.clamp(x, max=75)
            #print("clamp_x",torch.max(x))
            x = torch.expm1(x)
            #print(torch.max(x))
            x = x.mul(self.e[i])
            x = torch.clamp(x, max=75,min=0)
            
            min_values = torch.min(x, dim=0).values
            # # 各列の最大値 (dim=0 は列方向)
            max_values = torch.max(x, dim=0).values
            # Min-Max スケーリング
            x = (x - min_values) / ((max_values - min_values) + 1e-8)
            x = torch.tanh(x)
            #print(torch.max(x))

            #x = x.exp().mul(self.e[i])
            #print(x)
            #print("x",torch.isnan(x).sum())


   

            x = self.persona[:,i]*x
            #print(x[x<0])
            #print(x[x>1.])
           
            #print(self.persona)
            #m = nn.ReLU(inplace=False)
            #x = m(x)
            #print(int(i),x)
            probability += x
        #print(probability)
   
        
   

        return probability, next_feat, feat_prob

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
