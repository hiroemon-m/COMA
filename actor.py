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
    def __init__(self, T, e, r,q, w,persona) -> None:
        super().__init__()
        self.T = nn.Parameter(T.clone().detach().requires_grad_(True))
        self.e = nn.Parameter(e.clone().detach().requires_grad_(True))
        self.r = nn.Parameter(r.clone().detach().to(device).requires_grad_(True))
        self.q = nn.Parameter(q.clone().detach().to(device).requires_grad_(True))
        self.W = nn.Parameter(w.clone().detach().to(device).requires_grad_(True))
        self.persona = persona

    def calc_ration(self,attributes, edges,persona,time):
        calc_policy = torch.empty(len(persona[0][0]),32,32)
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)

        for i in range(len(persona[0][0])):
        
            tmp_tensor = self.W[time][i] * torch.matmul(edges, attributes)
            r = self.r[time][i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * self.q[time][i]
            feat_tanh = torch.tanh(feat)
            feat_tanh = torch.clamp(feat_tanh,min=0)
            feat_ber = feat_tanh.bernoulli()
            x = torch.mm(feat_ber, feat_ber.t())
            x = x.div(self.T[time][i]+1e-8)
            x = torch.clamp(x, max=79)
            x = torch.exp(x)
            x = x*self.e[time][i]
        

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
        self,attributes, edges,time
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attr= 0

        #print(self.persona[:,1])
        #print(self.persona)
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        #edges = (edges > 0).float().to(device)

        probability = 0
        for i in range(len(self.persona[0][0])):
 
            tmp_tensor = self.W[time][i] * torch.matmul(edges, attributes)
            r = self.r[time][i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * self.q[time][i]
            #print("feat",feat)
            feat_tanh = torch.tanh(feat)
            feat_tanh = torch.clamp(feat_tanh,min=0)
            feat_ber = feat_tanh
            x = torch.mm(feat_ber, feat_ber.t())
            x = x.div(self.T[time][i]+1e-8)
            x = torch.clamp(x, max=79)
            x = torch.exp(x)
            x = x*self.e[time][i]

            #print("nan",x,torch.isnan(x).sum())

            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)
     
            x = torch.tanh(x)
            #print("nan1",torch.isnan(x).sum())


            x = self.persona[time][:,i]*x
            attr = attr + self.persona[time][:,i].view(32,-1)*feat
    
            #print("nan2",torch.isnan(x).sum())
    
            probability =  torch.clamp(probability + x ,min=0,max=1)
            #属性の方策
        attr_tanh = torch.tanh(attr)
        attr_tanh = torch.clamp(attr_tanh,min=0)
        attr_ber = attr_tanh.bernoulli()
            #print("Feat",feat_ber)

            #feat = torch.where(feat>0,1,feat)
            #feat = torch.where(feat>1,1,feat)

        return probability, attr_ber




    def predict(self,attributes, edges,time):
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        #edges = (edges > 0).float().to(device)
        probability = 0
        attr = 0

            
        for i in range(len(self.persona[0][0])):
    
            tmp_tensor = self.W[time][i] * torch.matmul(edges, attributes)
            r = self.r[time][i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * self.q[time][i]
            feat_tanh = torch.tanh(feat)
            feat_tanh = torch.clamp(feat_tanh,min=0)
            #feat_ber = feat_tanh.bernoulli()
            feat_ber = feat_tanh 
            x = torch.mm(feat_ber, feat_ber.t())
                #print("feat",feat)
            x = x.div(self.T[time][i]+1e-8)
            #x = torch.clamp(x, max=75)
            #print("nan",x[x=="Nan"].sum())
            #print(x)
            #x = torch.exp(x)
                #expm1
                #print(torch.max(x))
            x = torch.clamp(x, max=79)
            x = torch.exp(x)
            x = x*self.e[time][i]

                # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)
            x = torch.tanh(x)
            x = self.persona[time][:,i]*x
            attr = attr + self.persona[time][:,i].view(-1,1)*feat

            #print(y)
            probability =  torch.clamp(probability + x ,min=0,max=1)
        #probability_tensor =  probability.clone()
        #print("pr",probability)
        #print(probability_tensor)
                        #属性の調整

        attr_tanh = torch.tanh(attr)
        attr_tanh = torch.clamp(attr_tanh,min=0)
        attr_ber = attr_tanh.bernoulli()

  
            #feat = torch.where(feat>0,1,feat)
            #feat = torch.where(feat>1,1,feat)


    

        return probability, attr_ber

    def test(self,edges,attributes,time):
        edges_float = edges.float()
        edge_index = edges_float > 0
        edges =edge_index.float().to(device)
        #edges = (edges > 0).float().to(device)
        probability = 0
        attr = 0
            
        for i in range(len(self.persona[0][0])):
    
            tmp_tensor = self.W[time][i] * torch.matmul(edges, attributes)
            r = self.r[time][i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * self.q[time][i]
            feat_tanh = torch.tanh(feat)
            feat_tanh = torch.clamp(feat_tanh,min=0)
            feat_ber = feat_tanh.bernoulli()
            x = torch.mm(feat_ber, feat_ber.t())
            x = x.div(self.T[time][i]+1e-8)
            #x = torch.clamp(x, max=75)
            #print("nan",x[x=="Nan"].sum())
            #print(x)
            #x = torch.exp(x)
                #expm1
                #print(torch.max(x))
            x = torch.clamp(x, max=79)
            x = torch.exp(x)
            x = x*self.e[time][i]
  


                # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)
            x = torch.tanh(x)
            x = self.persona[time][:,i]*x
            attr = attr + self.persona[time][:,i].view(-1,1)*feat

            #print(y)
            probability =  torch.clamp(probability + x ,min=0,max=1)
        #probability_tensor =  probability.clone()
        #print("pr",probability)
        #print(probability_tensor)
                        #属性の調整
        
        
  
 
        attr_tanh = torch.tanh(attr)
        attr_tanh = torch.clamp(attr_tanh,min=0)
        attr_ber = attr_tanh.bernoulli()

  
            #feat = torch.where(feat>0,1,feat)
            #feat = torch.where(feat>1,1,feat)

    

        return probability, attr_tanh,attr_ber




