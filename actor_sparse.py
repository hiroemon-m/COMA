# Standard Library
from typing import Tuple

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import gc

# First Party Library
import math
import config
import csv
import time
from memory_profiler import profile

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

  
    def calc_ration(self,attributes, edges,persona,sparse_size):
        a = time.time()

        calc_policy = [[[None] for _ in range(len(persona[0][0]))] for _ in range(5)]
        # インプレース操作を回避するために新しい変数を使用して新しいテンソルを作成

        agent_num = len(persona[0])
        # 繰り返し回数と新しいサイズを設定
        repeat_time = agent_num
        size = (agent_num,sparse_size)
        new_size = (size[0] , size[1])
        attr_ber = torch.empty(len(attributes), len(attributes[0]), 2)
        edge_ber = torch.empty(len(edges), len(edges[0]), 2)

        for i in range(len(persona[0][0])):

            feat = attributes.detach().clone()
            edge = edges.detach().clone()
            for t in range(5):

                # 列方向の合計を計算
                trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
                #sum_feat = torch.sparse.sum(feat, dim=0)
                #非ゼロ数の計算
                #num_nonzero = sum_feat._nnz()
                #非ゼロ要素のインデックスの計算
                #trend_indices = sum_feat.indices()
                #インデックスに基準行の追加
                #trend_indices = torch.cat([torch.tensor([[0 for _ in range(num_nonzero)]]), trend_indices], dim=0)
                #値のtensorの作成
                #values = torch.tensor([1] * num_nonzero)

                # インデックスの複製と新しいインデックスの生成
                #indices_repeated = trend_indices.clone()

                #for k in range(1, repeat_time):
                #    diff_indices = trend_indices.clone()
                #    diff_indices[0] = diff_indices[0] + k
                #    indices_repeated = torch.cat([indices_repeated, diff_indices], dim=1)
                # 繰り返したCOOを作成
                #trend = torch.sparse_coo_tensor(indices_repeated, values.repeat(repeat_time), new_size)
                trend = trend.to_sparse()

                feat_prob = torch.empty(len(feat),len(feat[0]),2)
                #tmp_tensor = self.W[i] * torch.sparse.spmm(edges, attributes) + trend

                tmp_tensor = torch.sparse.mm(edge,feat)*self.W[i] + trend
                r = self.r[i]
                feat = r * feat + tmp_tensor * (1-r)
                feat = feat.to_dense()
                feat_tanh = torch.tanh(feat)
                feat_prob[:,:,0] = 10 - feat_tanh * 10
                feat_prob[:,:,1] = feat_tanh * 10
                feat_action= self.gumbel_softmax(feat_prob,hard=True)
                del feat_prob
                feat = feat_action[:,:,1]
                norm = feat.norm(dim=1)[:, None] + 1e-8
                feat_norm = feat.div(norm)
                feat_sparse = feat_norm.to_sparse()
                x = torch.sparse.mm(feat_sparse, feat_sparse.t())

                x = x/(self.T[i]+1e-8)
                #x = torch.clamp(x, max=78)
                x = torch.exp(x.to_dense())
                x = x*self.e[i]   

                # Min-Max スケーリング
                min_values = torch.min(x, dim=0).values
                max_values = torch.max(x, dim=0).values
                x = (x - min_values) / ((max_values - min_values) + 1e-8)
                x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
          

    
                feat = feat.to_sparse()
                edge_prob =  x * 10
                edge_ber[:, :, 0] = 10 - edge_prob
                edge_ber[:, :, 1] = edge_prob
                edge = self.gumbel_softmax(edge_ber)[:, :, 1].to_sparse()

                calc_policy[t][i] = x.to_sparse()
                #print("{}feat{}".format(i,t),feat.indices())
                #print("e{}",edge.indices())

        del attr_ber,edge_ber
        b = time.time()

        return calc_policy

  


    def forward(self,attributes, edges,times,agent_num,sparse_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"trainで呼び出す"""
        a = time.time()

        edges_prob = 0

        trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
        trend = trend.to_sparse()

        for i in range(len(self.persona[0][0])):
        
            feat_prob = torch.empty(len(attributes),len(attributes[0]),2)     
            tmp_tensor = torch.sparse.mm(edges,attributes)*self.W[i] + trend
            r = self.r[i]
            feat = r * attributes + tmp_tensor * (1-r)
            feat = feat.to_dense()
            feat_tanh = torch.tanh(feat)
            feat_prob[:,:,0] = 10 - feat_tanh*10
            feat_prob[:,:,1] = feat_tanh*10
            feat= self.gumbel_softmax(feat_prob,hard=True)
            del feat_prob
            feat = feat[:,:,1]
            norm = feat.norm(dim=1)[:, None] + 1e-8
            feat = feat.div(norm)
            feat_sparse = feat.to_sparse()
            x = torch.sparse.mm(feat_sparse, feat_sparse.t())

            x = x/(self.T[i]+1e-8)
            #x = torch.clamp(x, max=78)
            x = torch.exp(x.to_dense())
            x = x*self.e[i] 

            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)

            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            x = self.persona[times][:,i]*x 
            edges_prob = edges_prob + x
            del feat, norm, feat_sparse, x  # メモリ解放
        edges_prob = (edges_prob + 1e-3).to_sparse()
        
        b = time.time()
        print("fw",b-a)

        return edges_prob



    #@profile    
    def predict(self, attributes, edges, times, agent_num, sparse_size):
        a = time.time()
        with torch.no_grad():
            attr_ber = torch.empty(len(attributes), len(attributes[0]), 2)
            edge_ber = torch.empty(len(edges), len(edges[0]), 2)
            edges_prob = 0
            attr_prob = 0
   
            trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
            trend = trend.to_sparse()

            for i in range(len(self.persona[0][0])):

                tmp_feat_prob = torch.empty(len(attributes), len(attributes[0]), 2)
                tmp_mm_result = torch.sparse.mm(edges, attributes) * self.W[i] + trend
         
                
                r = self.r[i]
                tmp_feat = r * attributes + tmp_mm_result * (1 - r)
                del tmp_mm_result
                
                tmp_feat = tmp_feat.to_dense()
                tmp_feat_tanh = torch.tanh(tmp_feat)
                tmp_feat_prob[:, :, 0] = 10 - (tmp_feat_tanh * 10)
                tmp_feat_prob[:, :, 1] = (tmp_feat_tanh * 10)
                tmp_feat_action = self.gumbel_softmax(tmp_feat_prob, hard=True)
                del tmp_feat_prob
                
                tmp_feat_action = tmp_feat_action[:, :, 1]
                tmp_norm = tmp_feat_action.norm(dim=1)[:, None] + 1e-8
                tmp_feat_action = tmp_feat_action.div(tmp_norm)
                tmp_feat_sparse = tmp_feat_action.to_sparse()
                tmp_x = torch.sparse.mm(tmp_feat_sparse, tmp_feat_sparse.t())
                del tmp_feat_sparse, tmp_feat_action

                tmp_x = tmp_x / (self.T[i] + 1e-8)
                tmp_x = torch.exp(tmp_x.to_dense())
                tmp_x = tmp_x * self.e[i] 

                tmp_min_values = torch.min(tmp_x, dim=0).values
                tmp_max_values = torch.max(tmp_x, dim=0).values
                tmp_x = (tmp_x - tmp_min_values) / ((tmp_max_values - tmp_min_values) + 1e-8)
                del tmp_min_values, tmp_max_values

                tmp_x = (1 - torch.exp(-tmp_x - tmp_x)) / (1 + torch.exp(-tmp_x - tmp_x))
                tmp_x = self.persona[times][:, i] * tmp_x
                edges_prob = edges_prob + tmp_x
                attr_prob = attr_prob + self.persona[times][:, i].view(-1, 1) * tmp_feat_tanh
                del tmp_feat, tmp_feat_tanh, tmp_x

            attr_prob = attr_prob * 10
            attr_ber[:, :, 0] = 10 - attr_prob
            attr_ber[:, :, 1] = attr_prob
            attr_action = self.gumbel_softmax(attr_ber)[:, :, 1].to_sparse()
            del attr_ber, attr_prob

            edge_prob = edges_prob * 10
            edge_ber[:, :, 0] = 10 - edge_prob
            edge_ber[:, :, 1] = edge_prob
            edge_action = self.gumbel_softmax(edge_ber)[:, :, 1].to_sparse()
            del edge_ber, edge_prob

            b = time.time()
            print("pr",b-a)

            return edges_prob, edge_action, attr_action

    

    def test(self,edges,attributes,times,agent_num,sparse_size):
        a = time.time()
        
      
        probability = 0
        attr_prob = 0
        attr_ber = torch.empty(len(attributes),len(attributes[0]),2)
        edge_ber = torch.empty(len(edges),len(edges[0]),2)
            
        trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
        trend = trend.to_sparse()

            
        for i in range(len(self.persona[0][0])):

    
            feat_prob = torch.empty(len(attributes),len(attributes[0]),2)     
            tmp_tensor = torch.sparse.mm(edges,attributes)*self.W[i] + trend
            r = self.r[i]
            feat = r * attributes + tmp_tensor * (1-r)
            feat = feat.to_dense()
            feat_tanh = torch.tanh(feat)

            feat_prob[:,:,0] = 10 - (feat_tanh*10)
            feat_prob[:,:,1] = feat_tanh*10
            feat= self.gumbel_softmax(feat_prob,hard=True)
            del feat_prob
            feat = feat[:,:,1]
            norm = feat.norm(dim=1)[:, None] + 1e-8
            feat = feat.div(norm)
            feat_sparse = feat.to_sparse()
            x = torch.sparse.mm(feat_sparse, feat_sparse.t())

            del feat, norm, feat_sparse 

            x = x/(self.T[i]+1e-8)
            #x = torch.clamp(x, max=78)
            x = torch.exp(x.to_dense())
            x = x*self.e[i] 

            # Min-Max スケーリング
            min_values = torch.min(x, dim=0).values
            max_values = torch.max(x, dim=0).values
            x = (x - min_values) / ((max_values - min_values) + 1e-8)

            x = (1 - torch.exp(-x-x))/(1 + torch.exp(-x-x))
            x = self.persona[times][:,i]*x
            probability = probability + x
            attr_prob = attr_prob + self.persona[times][:,i].view(-1,1)*feat_tanh

            del x, min_values, max_values, feat_tanh 


        #属性の調整

        attr_ber[:,:,0] = 10 - attr_prob*10
        attr_ber[:,:,1] = attr_prob*10
        attr_action= self.gumbel_softmax(attr_ber)[:,:,1].to_sparse()

        edge_prob = probability*10
        edge_ber[:,:,0] = 10 - edge_prob
        edge_ber[:,:,1] = edge_prob
        edge_action= self.gumbel_softmax(edge_ber)[:,:,1].to_sparse()
        print("属性",attr_prob[0])
        print(attr_prob[1])

        del attr_ber,edge_ber

        b = time.time()
        print("test",b-a)

        return edge_action,probability,attr_prob,attr_action




