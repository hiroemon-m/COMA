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
        logits_indeices = logits.indices()
        y = logits + gumbel_noise[logits_indeices]
        y = y/tau
        
        y_softmax = torch.sparse_coo_tensor(y.indices(),F.softmax(y.values()),y.size())
        print("ysoftmax",y_softmax)
        return y_softmax

    def gumbel_softmax(self,logits,hard=False):
        """Gumbel-Softmaxサンプリング、ハードサンプルもサポート"""
        y = self.gumbel_softmax_sample(logits, self.temperature)

        if hard:
            # ハードサンプル：one-hotにするが、勾配はソフトサンプルに基づく
            shape = y.size()
            y = y.to_dense()
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
       # repeat_time = agent_num
        #size = (agent_num,sparse_size)
        #new_size = (size[0] , size[1])
        #attr_ber = torch.empty(len(attributes), len(attributes[0]), 2)
        #edge_ber = torch.empty(len(edges), len(edges[0]), 2)

        for i in range(len(persona[0][0])):

            feat = attributes.detach().clone()
            edge = edges.detach().clone()
            for t in range(5):

                # 列方向の合計を計算
                #trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
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
                #trend = trend.to_sparse()

                #feat_prob = torch.empty(len(feat),len(feat[0]),2)
                #tmp_tensor = self.W[i] * torch.sparse.spmm(edges, attributes) + trend

                #tmp_tensor = torch.sparse.mm(edge,feat)*self.W[i] + trend
                tmp_tensor = torch.sparse.mm(edge,feat)*self.W[i] 
                r = self.r[i]
                tmp_feat = r * feat + tmp_tensor * (1-r)
                #feat = feat.to_dense()
                #feat_tanh = torch.tanh(feat)
                #feat_prob[:,:,0] = 10 - feat_tanh * 10
                #feat_prob[:,:,1] = feat_tanh * 10
                #feat_action= self.gumbel_softmax(feat_prob,hard=True)
                tmp_feat_action = tmp_feat

   


                #feat = feat_action[:,:,1]
                #norm = feat.norm(dim=1)[:, None] + 1e-8
                #feat_norm = feat.div(norm)
                #feat_sparse = feat_norm.to_sparse()

                tmp_feat_sparse = tmp_feat_action.to_sparse()
                tmp_x = torch.sparse.mm(tmp_feat_sparse, tmp_feat_sparse.t())
                del tmp_feat_sparse


                tmp_x = tmp_x / (self.T[i] + 1e-8)
                tmp_x_value = tmp_x.values()
                tmp_exp = torch.exp(tmp_x_value)
                tmp_x = torch.sparse_coo_tensor(tmp_x.indices(),tmp_exp,tmp_x.size())
                tmp_x_e = tmp_x * self.e[i] 

                tmp_x_e= tmp_x_e.coalesce()
                tmp_x_e_value = tmp_x_e.values()
                tmp_x_e_value = (1 - torch.exp(-tmp_x_e_value-tmp_x_e_value))/(1 + torch.exp(-tmp_x_e_value-tmp_x_e_value))

                non_zero_indices = tmp_x_e_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
                filtered_indices = tmp_x_e.indices()[:, non_zero_indices]  # 0でないインデックスのみ
                filtered_values = tmp_x_e_value[non_zero_indices]  # 0でない値のみ 
                tmp_coo = torch.sparse_coo_tensor(filtered_indices,filtered_values,tmp_x_e.size())


                tmp_coo = self.persona[t][:, i] * tmp_coo
                if i ==0 :
                    edges_prob =  tmp_coo
                edges_prob += tmp_coo*self.persona[t][:, i].view(-1, 1)


                del tmp_feat, tmp_x
    
                

                calc_policy[t][i] = edges_prob
                #print("{}feat{}".format(i,t),feat.indices())
                #print("e{}",edge.indices())

       
        b = time.time()

        return calc_policy

  


    def forward(self,attributes, edges,times,agent_num,sparse_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"trainで呼び出す"""
        a = time.time()

        

        #trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
        #trend = trend.to_sparse()

        for i in range(len(self.persona[0][0])):
  

            tmp_feat_prob = torch.empty(len(attributes), len(attributes[0]), 2)
            tmp_mm_result = torch.sparse.mm(edges, attributes) * self.W[i] 


            
            r = self.r[i]
            tmp_feat = r * attributes + tmp_mm_result * (1 - r)
            del tmp_mm_result
            
            tmp_feat_action = tmp_feat.coalesce()
            print("3",tmp_feat_action)
            indices = tmp_feat_action.indices()
            values = tmp_feat_action.values()
            size = tmp_feat_action.size()
            #/////////////////////////////////#
            row_indices = indices[0]                       # 行インデックス
            squared_values = values ** 2                   # 各値の平方
            row_sums = torch.zeros(size[0])                # 行ごとの平方和を格納
            row_sums.index_add_(0, row_indices, squared_values)  # 行ごとに値を加算
            l2_norms = torch.sqrt(row_sums)                # 平方根を取る

            # 正規化：各非ゼロ要素を行ごとの L2 ノルムで割る
            normalized_values = values / l2_norms[row_indices]

            # 正規化されたスパーステンソルを作成
            normalized_tmp_feat = torch.sparse_coo_tensor(indices, normalized_values, size)
            print("3.5",normalized_tmp_feat)
            tmp_x = torch.sparse.mm(normalized_tmp_feat, normalized_tmp_feat.t()).coalesce()
            #mask = tmp_x.values() > 0.5  # 条件を満たす値のマスク
            #new_values = torch.where(mask, tmp_x.values(), torch.tensor(-5))  # 値を変更

            # 新しいスパーステンソルを作成
            comp_tmp_x = torch.sparse_coo_tensor(tmp_x.indices(), tmp_x, tmp_x.size()).coalesce()

            comp_tmp_x = comp_tmp_x / (self.T[i] + 1e-8)
            tmp_x_value = comp_tmp_x.values()
            tmp_exp = torch.exp(tmp_x_value)
            comp_tmp_x = torch.sparse_coo_tensor(comp_tmp_x.indices(),tmp_exp,comp_tmp_x.size())
            tmp_x_e = comp_tmp_x * self.e[i] 
            print("5",tmp_x_e)


            tmp_x_e= tmp_x_e.coalesce()
            size = tmp_x_e.size()                            # サイズ
            e_value = tmp_x_e.values()
            indices = tmp_x_e.indices()

            # 非ゼロ要素の min-max を計算
            min_value = 0   # 最小値
            max_value = torch.max(e_value)   # 最大値
            print("max",max_value)

            # Min-Max スケーリング
            scaled_e_values = (e_value - min_value) / ((max_value - min_value) + 1e-8)
            print("5",scaled_e_values)


            # スケーリングしたスパーステンソルを作成
            tmp_min_max = torch.sparse_coo_tensor(indices, scaled_e_values, size)
            tmp_min_max = tmp_min_max.coalesce()
            
            tmp_x_e_value = tmp_min_max.values()
            tmp_x_e_value = (1 - torch.exp(-tmp_x_e_value-tmp_x_e_value))/(1 + torch.exp(-tmp_x_e_value-tmp_x_e_value))
            print(tmp_x_e_value)

            non_zero_indices = tmp_x_e_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
            filtered_indices = tmp_x_e.indices()[:, non_zero_indices]  # 0でないインデックスのみ
            filtered_values = tmp_x_e_value[non_zero_indices]  # 0でない値のみ 
            tmp_coo = torch.sparse_coo_tensor(filtered_indices,filtered_values,tmp_min_max.size())

            tmp_coo = self.persona[times][:, i] * tmp_coo
            if i ==0 :
                edges_prob =  tmp_coo
            edges_prob += tmp_coo*self.persona[times][:, i].view(-1, 1)

            del tmp_feat, tmp_x
       

        edges_prob_size = edges_prob.size()
        b = time.time()
        print("fw",b-a)
  
        edges_prob = edges_prob.coalesce()

        #non_zero_indices = edges_prob.values().nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
        #filtered_indices = edges_prob.indices()[:, non_zero_indices]  # 0でないインデックスのみ
        #filtered_values = edges_prob.values()[non_zero_indices]  # 0でない値のみ 
        #edges_prob = torch.sparse_coo_tensor(filtered_indices,filtered_values,edges_prob_size).coalesce()

        #print("EEEE",torch.sum(edges_prob),edges_prob.values(),edges_prob.indices())
        print("EEEE",torch.sum(edges_prob),edges_prob.size())

        return edges_prob



    #@profile    
    def predict(self, attributes, edges, times, agent_num, sparse_size):

        a = time.time()
        with torch.no_grad():
            #attr_ber = torch.empty(len(attributes), len(attributes[0]), 2)
            #edge_ber = torch.empty(len(edges), len(edges[0]), 2)
            #trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
            #trend = trend.to_sparse()
            print(attributes)
            print(edges)

            for i in range(len(self.persona[0][0])):
                #ここで値域を決める場合と、tanh前に入れる方法がある。
                policy_max = torch.tanh(self.e[i]*torch.exp(1/self.T[i]))
                policy_min = torch.tanh(self.e[i]*torch.exp(0/self.T[i]))
       

                print("1",policy_max,policy_min)
                tmp_feat_prob = torch.empty(len(attributes), len(attributes[0]), 2)
                #tmp_mm_result = torch.sparse.mm(edges, attributes) * self.W[i] + trend
                tmp_mm_result = torch.sparse.mm(edges, attributes) * self.W[i] 
                print("2",tmp_mm_result)
        
                r = self.r[i]
                tmp_feat = r * attributes + tmp_mm_result * (1 - r)
                del tmp_mm_result

                #t+1の属性値を確率に変換
                tmp_feat_action = tmp_feat.coalesce()
                attr_indcies = tmp_feat_action.indices()
                attr_values = tmp_feat_action.values()
                attr_size = tmp_feat_action.size()
                attr_sigmoid = torch.sigmoid(attr_values)
                feat_sigmoid_prob = torch.sparse_coo_tensor(attr_indcies,attr_sigmoid,attr_size).coalesce()

                del attr_indcies,attr_values,attr_size,attr_sigmoid

                #確率から行動に変換
                #feat_sigmoid_indcies = feat_sigmoid_prob.indices()
                #feat_sigmoid_valuse = feat_sigmoid_prob.values()
                #feat_sigmoid_size = feat_sigmoid_prob.size()
                #feat_action = torch.bernoulli(feat_sigmoid_valuse)
                #tmp_feat_action = torch.sparse_coo_tensor(feat_sigmoid_indcies,feat_action,feat_sigmoid_size).coalesce()
             
                print("3",tmp_feat_action)
 
                # 行ごとの L2 ノルムを計算
                indices = tmp_feat_action.indices()
                values = tmp_feat_action.values()
                size = tmp_feat_action.size()
                #/////////////////////////////////#
                row_indices = indices[0]                       # 行インデックス
                squared_values = values ** 2                   # 各値の平方
                row_sums = torch.zeros(size[0])                # 行ごとの平方和を格納
                row_sums.index_add_(0, row_indices, squared_values)  # 行ごとに値を加算
                l2_norms = torch.sqrt(row_sums)                # 平方根を取る

                # 正規化：各非ゼロ要素を行ごとの L2 ノルムで割る
                normalized_values = values / l2_norms[row_indices]

                # 正規化されたスパーステンソルを作成
                normalized_tmp_feat = torch.sparse_coo_tensor(indices, normalized_values, size)
        
                #/////////////////////////////////#
                #--------------------------#
                #l2_norm = torch.sqrt(torch.sum(values ** 2))

                # 非ゼロ要素を L2 ノルムで割る（正規化）
                #normalized_values = values / l2_norm

                # 正規化されたスパーステンソルを作成
                #normalized_tmp_feat = torch.sparse_coo_tensor(indices, normalized_values, size)
                #--------------------------#

                print("3.5",normalized_tmp_feat)

    
                tmp_x = torch.sparse.mm(normalized_tmp_feat, normalized_tmp_feat.t()).coalesce()
                #
                #mask = tmp_x.values() > 0.5  # 条件を満たす値のマスク
                #new_values = torch.where(mask, tmp_x.values(), torch.tensor(-5))  # 値を変更

                # 新しいスパーステンソルを作成
                comp_tmp_x = tmp_x
                del normalized_tmp_feat,tmp_x
                print("4",comp_tmp_x)

                comp_tmp_x = comp_tmp_x / (self.T[i] + 1e-8)
                tmp_x_value = comp_tmp_x.values()
                tmp_exp = torch.exp(tmp_x_value)
                comp_tmp_x = torch.sparse_coo_tensor(comp_tmp_x.indices(),tmp_exp,comp_tmp_x.size())
                tmp_x_e = comp_tmp_x * self.e[i] 
                print("5",tmp_x_e)

                #tmp_min_values = torch.min(tmp_x.to_dense(), dim=0)
                #tmp_max_values = torch.max(tmp_x.to_dense(), dim=0)
                #tmp_x = (tmp_x - tmp_min_values) / ((tmp_max_values - tmp_min_values) + 1e-8)
                #del tmp_min_values, tmp_max_values
                #print("6",tmp_x)
                tmp_x_e = tmp_x_e.coalesce()                          # サイズ
                #size = tmp_x_e.size()                            # サイズ
                #e_value = tmp_x_e.values()
                #indices = tmp_x_e.indices()

                # 非ゼロ要素の min-max を計算
                #min_value = 0   # 最小値
                #max_value = torch.max(e_value)   # 最大値
                #print("max",max_value)

                # Min-Max スケーリング
                #scaled_e_values = (e_value - min_value) / ((max_value - min_value) + 1e-8)
                #print("5",scaled_e_values)


                # スケーリングしたスパーステンソルを作成
                #tmp_min_max = torch.sparse_coo_tensor(indices, scaled_e_values, size)
                #tmp_min_max = tmp_min_max.coalesce()
                
                #tmp_x_e_value = tmp_min_max.values()
                tmp_x_e_value = tmp_x_e.values()
                tmp_x_e_value = (1 - torch.exp(-tmp_x_e_value-tmp_x_e_value))/(1 + torch.exp(-tmp_x_e_value-tmp_x_e_value))
                tmp_x_e_value = ((tmp_x_e_value-policy_min)/(policy_max - policy_min))
                print(tmp_x_e_value)

                non_zero_indices = tmp_x_e_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
                filtered_indices = tmp_x_e.indices()[:, non_zero_indices]  # 0でないインデックスのみ
                filtered_values = tmp_x_e_value[non_zero_indices]  # 0でない値のみ 
                mask = filtered_indices[0] != filtered_indices[1]  # 行番号と列番号が異なる要素だけを選択
                # マスクを使って新しいインデックスと値をフィルタリング
                new_indices = filtered_indices[:, mask]
                new_values = filtered_values[mask]
                tmp_coo = torch.sparse_coo_tensor(new_indices,new_values,tmp_x_e.size())

                tmp_coo = self.persona[times][:, i] * tmp_coo
                if i ==0 :
                    edges_prob =  tmp_coo
                edges_prob += tmp_coo*self.persona[times][:, i].view(-1, 1)
                #attr_prob = attr_prob + self.persona[times][:, i].view(-1, 1) * tmp_feat_tanh
                print("ss",tmp_feat_action.size(),self.persona[times][:, i].view(-1, 1).size())
                if i ==0:
                    attr_prob = tmp_feat_action*self.persona[times][:, i].view(-1, 1)
                attr_prob = attr_prob + tmp_feat_action*self.persona[times][:, i].view(-1, 1)
                print("pe",self.persona[times][:, i])
                
                del tmp_feat
                print("7",attr_prob,edges_prob.coalesce())

            
            attr_indces = attr_prob.indices()
            attr_size = attr_prob.size()
            attr_prob_values = attr_prob.values()
            attr_value = torch.bernoulli(torch.sigmoid(attr_prob_values))

            # 0でない値のインデックスをフィルタリング
            non_zero_indices = attr_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
            filtered_indices = attr_indces[:, non_zero_indices]  # 0でないインデックスのみ
            filtered_values = attr_prob_values[non_zero_indices]  # 0でない値のみ 
            attr_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,attr_size)

            #del attr_ber, attr_prob
            gc.collect()
            print("done attr",attr_action)

            edges_prob = edges_prob.coalesce()
            edge_indces = edges_prob.indices()
            edge_size = edges_prob.size()
            print("e_prob",edges_prob.indices())
            print("e_prob",edges_prob.values())
            edge_value = torch.bernoulli(torch.tanh(edges_prob.values()))

           

            # 0でない値のインデックスをフィルタリング
            non_zero_indices = edge_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
            filtered_indices = edge_indces[:, non_zero_indices]  # 0でないインデックスのみ
            filtered_values = edge_value[non_zero_indices]  # 0でない値のみ 
            edge_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,edge_size).coalesce()
            print("e_action",edge_action)
            

          
            gc.collect()

            b = time.time()
            print(attr_action)
            print("pr",b-a,torch.sum(attr_action),edge_size)

            return edges_prob, edge_action, attr_action

    

    def test(self,edges,attributes,times,agent_num,sparse_size):
        a = time.time()
        
      
        #probability = 0
        #attr_prob = 0
        #attr_ber = torch.empty(len(attributes),len(attributes[0]),2)
        #edge_ber = torch.empty(len(edges),len(edges[0]),2)
            
        #trend = (torch.sum(attributes.to_dense(),dim=0)>0).repeat(self.agent_num,1)
        #trend = trend.to_sparse()

            
        for i in range(len(self.persona[0][0])):

            tmp_feat_prob = torch.empty(len(attributes), len(attributes[0]), 2)
            tmp_mm_result = torch.sparse.mm(edges, attributes) * self.W[i] 

            r = self.r[i]
            tmp_feat = r * attributes + tmp_mm_result * (1 - r)

            del tmp_mm_result
            
            tmp_feat_action = tmp_feat
            tmp_feat_sparse = tmp_feat_action.to_sparse()
            tmp_x = torch.sparse.mm(tmp_feat_sparse, tmp_feat_sparse.t())
            del tmp_feat_sparse
   
            tmp_x = tmp_x / (self.T[i] + 1e-8)
            tmp_x_value = tmp_x.values()
            tmp_exp = torch.exp(tmp_x_value)
            tmp_x = torch.sparse_coo_tensor(tmp_x.indices(),tmp_exp,tmp_x.size())
            tmp_x_e = tmp_x * self.e[i] 
            tmp_x_e= tmp_x_e.coalesce()
            tmp_x_e_value = tmp_x_e.values()
            tmp_x_e_value = (1 - torch.exp(-tmp_x_e_value-tmp_x_e_value))/(1 + torch.exp(-tmp_x_e_value-tmp_x_e_value))

            non_zero_indices = tmp_x_e_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
            filtered_indices = tmp_x_e.indices()[:, non_zero_indices]  # 0でないインデックスのみ
            filtered_values = tmp_x_e_value[non_zero_indices]  # 0でない値のみ 
            tmp_coo = torch.sparse_coo_tensor(filtered_indices,filtered_values,tmp_x_e.size())

            tmp_coo = self.persona[times][:, i] * tmp_coo

            if i ==0 :
                edges_prob =  tmp_coo
            edges_prob += tmp_coo*self.persona[times][:, i].view(-1, 1)
           
            if i ==0:
                attr_prob = tmp_feat_action*self.persona[times][:, i].view(-1, 1)
            attr_prob = attr_prob + tmp_feat_action*self.persona[times][:, i].view(-1, 1)           
            
            del tmp_feat, tmp_x

        attr_prob = attr_prob.coalesce()/len(self.persona[0][0])
        attr_indces = attr_prob.indices()
        attr_size = attr_prob.size()
        attr_value = torch.where(attr_prob.values() >= 0.005,
                                torch.tensor(1.0, requires_grad=True), 
                                torch.tensor(0.0, requires_grad=True)
                                )
        non_zero_indices = attr_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
        filtered_indices = attr_indces[:, non_zero_indices]  # 0でないインデックスのみ
        filtered_values = attr_value[non_zero_indices]  # 0でない値のみ 
        attr_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,attr_size)

        #del attr_ber, attr_prob
        gc.collect()

        edges_prob = edges_prob.coalesce()
        edge_indces = edges_prob.indices()
        edge_size = edges_prob.size()
        edge_value = torch.where(edges_prob.values() >= 0.5,
                                torch.tensor(1.0, requires_grad=True), 
                                torch.tensor(0.0, requires_grad=True)
                                )

        non_zero_indices = edge_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
        filtered_indices = edge_indces[:, non_zero_indices]  # 0でないインデックスのみ
        filtered_values = edge_value[non_zero_indices]  # 0でない値のみ 
        edge_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,edge_size)





        b = time.time()
        print("test",b-a)

        return edge_action,edges_prob,attr_prob,attr_action




