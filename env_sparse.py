# Standard Library
import math

# Third Party Library
import networkx as nx
import torch.nn as nn
from torchviz import make_dot


import numpy as np
import scipy
import torch

# First Party Library
import config

device = config.select_device

"""2hop先の隣接行列の対角成分を取り除く"""
def remove_diagonal(data):

    filtered_indices = data.indices()
    filtered_values = data.values()
    mask = filtered_indices[0] != filtered_indices[1]  # 行番号と列番号が異なる要素だけを選択
    # マスクを使って新しいインデックスと値をフィルタリング
    new_indices = filtered_indices[:, mask]
    new_values = filtered_values[mask]
    remove_diagonal_sparse = torch.sparse_coo_tensor(new_indices,new_values,data.size()).coalesce()

    return remove_diagonal_sparse
            


def norm_func(x):
    x_indices = x.indices()
    x_values = x.values().requires_grad_(True)
    x_size = x.size()
    x_row_indices = x_indices[0]            # 行インデックス
    x_squared_values = x_values ** 2                # 各値の平方
    
    # in-place操作を避けるため、新しいテンソルを作��
    x_row_sums = torch.zeros(x_size[0], requires_grad=True)
    x_row_sums = x_row_sums.scatter_add(0, x_row_indices, x_squared_values)  # in-place操作を避ける
    
    x_l2_norms = torch.sqrt(x_row_sums)                # 平方根を取る
    x_normalized_values = x_values / x_l2_norms[x_row_indices] # 正規化
    
    # スパーステンソルを作成
    normalized_x_feat = torch.sparse_coo_tensor(
        x_indices, 
        x_normalized_values, 
        x_size,
        requires_grad=True
    ).coalesce()

    return normalized_x_feat



"""隣接行列の類似度計算"""
def adj_sim1(adj, feat):
    # 隣接行列
    A_sparse = adj.coalesce()
    row_indices, col_indices = A_sparse.indices()
    values_A = A_sparse.values()
    
    # 属性値行列
    B_sparse = feat.coalesce()
    B_indices = B_sparse.indices()
    B_values = B_sparse.values()

    # 非ゼロインデックスに対応する行列積を計算
    result_indices = []
    result_values = []

    for i, j in zip(row_indices, col_indices):
        # B の行要素と列要素を取得
        row_mask = B_indices[0] == i.item()
        col_mask = B_indices[0] == j.item()
        
        row_b = B_values[row_mask]
        col_b = B_values[col_mask]

        # 行要素と列要素のインデックスを比較して共��部分を計算
        if len(row_b) > 0 and len(col_b) > 0:
            # サイズを合わせるために行列積を取る
            row_indices_b = B_indices[1][row_mask]
            col_indices_b = B_indices[1][col_mask]

            # 共通する列要素を抽出
            common_indices = torch.isin(row_indices_b, col_indices_b)
            filtered_row_b = row_b[common_indices]

            common_indices = torch.isin(col_indices_b, row_indices_b)
            filtered_col_b = col_b[common_indices]

            # 要素が一致した場合のみ積を計算
            if len(filtered_row_b) > 0 and len(filtered_col_b) > 0:
                result_indices.append([i.item(), j.item()])
                result_values.append(torch.sum(filtered_row_b * filtered_col_b).item())

    # スパース結果を作成
    result_indices = torch.tensor(result_indices, dtype=torch.long).t()  # 転置して形状を整える
    result_values = torch.tensor(result_values, dtype=torch.float32)
    result_sparse = torch.sparse_coo_tensor(
        result_indices,
        result_values.requires_grad_(True),
        size=adj.size(),
        requires_grad=True
    )

    return result_sparse

def sparse_hadamard_product(adj, similarity):
    """
    スパーステンソルのアダマール積を計算。共通インデックスのみを効率的に扱う。
    """
    # インデックスと値を取得
    indices_a, values_a = adj._indices(), adj._values()
    indices_b, values_b = similarity._indices(), similarity._values()

    # インデックスをキーとして辞書を作成
    index_map_a = {tuple(idx.tolist()): i for i, idx in enumerate(indices_a.t())}
    index_map_b = {tuple(idx.tolist()): i for i, idx in enumerate(indices_b.t())}

    # 共通インデックスを特定
    common_keys = set(index_map_a.keys()).intersection(index_map_b.keys())

    # 共通インデックスの値を計算
    common_indices = torch.tensor(list(common_keys), dtype=torch.long).t()
    common_values = torch.tensor(
        [
            (values_a[index_map_a[key]] * values_b[index_map_b[key]]).requires_grad_(True)
            for key in common_keys
        ],
        dtype=values_a.dtype,
    )

    # 新しいスパーステンソルを作成
    result = torch.sparse_coo_tensor(common_indices, common_values, adj.size(),requires_grad=True).coalesce()
    return result


def adj_sim_self(adj, feat):
    """gammaの計算"""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    neigh_feat = torch.sparse.mm(adj_sparse, feat_sparse)
    similarity = torch.sparse.mm(neigh_feat, neigh_feat.t())
    print("ガンマsimilarity",similarity)

    return similarity
    
def adj_sim(adj, feat):
    """隣接行列の類似度計算."""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    #L２ノルムの計算
    normalized_feat = norm_func(feat)
    #ノードの類似度の計算
    similarity = torch.sparse.mm(normalized_feat, normalized_feat.t())
    neigh_similarity = sparse_hadamard_product(adj_sparse, similarity)
    print("アルファsimilarity",neigh_similarity)

    return neigh_similarity



class Env(nn.Module):
    def __init__(self, agent_num,edge, feature,alpha,beta,gamma,persona) -> None:
        #t=4での状態に初期��
        super().__init__()
        self.agent_num = agent_num
        self.edges = edge
        self.feature = feature
        self.feature_t = self.feature.t()
        self.alpha = nn.Parameter(alpha.clone().detach().requires_grad_(True))
        self.beta = nn.Parameter(beta.clone().detach().requires_grad_(True))
        self.gamma = nn.Parameter(gamma.clone().detach().requires_grad_(True))
        self.persona = persona.clone().detach().requires_grad_(False)
        
  

    def reset(self, edges, attributes,persona):
        self.edges = edges
        self.feature = attributes
        self.feature_t = self.feature.t()
        self.persona = persona.detach().clone().requires_grad_(True)

        return self.edges,self.feature


    """報酬を計算する"""
    def step(self,next_feature,next_action,time):
        
        persona_num = self.persona[time].size()[1]

        #impactを計算:(L2ノルム)^2
        old_feature = self.feature 
        new_feature = next_feature 
        diff_feature = torch.sub(old_feature,new_feature).coalesce() #属性値の差を求める
        impact_coo = adj_sim_self(next_action,diff_feature)
        impact_norm = impact_coo/(self.feature[0].size()[0])
        persona_gamma = torch.mm(self.persona[time],self.gamma.view(persona_num,1)) #gammaを計算
        reward_impact = impact_norm.multiply(persona_gamma).coalesce()

        #simlalityを計算
        normed_next_feature = norm_func(next_feature) #単位ベクトルに変更する
        similality_coo = adj_sim(next_action,normed_next_feature)
        persona_alpha = torch.mm(self.persona[time],self.alpha.view(self.persona[time].size()[1],1))
        reward_sim = similality_coo.multiply(persona_alpha).coalesce()

        #costを計算
        persona_beta = torch.mm(self.persona[time],self.beta.view(self.persona[time].size()[1],1))
        reward_cost = self.edges.multiply(persona_beta).to_sparse().coalesce()
        reward = reward_sim - reward_cost + reward_impact

        #edges,featureを更新
        self.edges = next_action
        self.feature = next_feature

        return reward
    
    def update_reward(self,next_feature,one_hop_action,time,alpha_all=None,beta_all=None,gamma_all=None):

        if time == 0:
            alpha_all = torch.mm(self.persona[time],self.alpha.view(-1,1)).to_sparse().clone().detach().requires_grad_(True)
            beta_all = torch.mm(self.persona[time],self.beta.view(-1,1)).to_sparse().clone().detach().requires_grad_(True)
            gamma_all = torch.mm(self.persona[time],self.gamma.view(-1,1)).to_sparse().clone().detach().requires_grad_(True)
        else:
            alpha_all = alpha_all.clone().detach().requires_grad_(True)
            beta_all = beta_all.clone().detach().requires_grad_(True)
            gamma_all = gamma_all.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([alpha_all,beta_all,gamma_all],lr=0.01)


       
        # 勾配追跡用のデバッグ出力を追加
        def debug_grad(name, tensor):
            if tensor.requires_grad:
                print(f"{name} requires_grad=True")
                if tensor.grad is not None:
                    print(f"{name} grad={tensor.grad}")
            else:
                print(f"{name} requires_grad=False")

        # impact計算の勾配追跡
        old_feature = self.feature.detach().clone().requires_grad_(True)
        new_feature = next_feature.detach().clone().requires_grad_(True)
        edge = self.edges.detach().clone().requires_grad_(True)
        diff_feature = torch.sub(old_feature, new_feature)
        debug_grad("diff_feature", diff_feature)
        
        impact_coo = adj_sim_self(one_hop_action, diff_feature).coalesce()

        impact_norm = impact_coo/(self.feature[0].size()[0])
        row_indices = impact_norm.indices()[0]
        row_values = impact_norm.values()
        row_sum = torch.zeros(impact_norm.size()[0]).scatter_add_(0, row_indices, row_values)
        impact_norm_sum = torch.sparse_coo_tensor(impact_norm.indices(),row_sum,impact_norm.size())
        print("impact_norm",impact_norm.size())
        print("gamma_all",gamma_all.size())
        
        reward_impact = impact_norm_sum.multiply(gamma_all).coalesce()
        debug_grad("reward_impact", reward_impact)

        # similarity計算の勾配追跡
        similality_coo = adj_sim(one_hop_action, next_feature)
        debug_grad("similality_coo", similality_coo)
        
        reward_sim = similality_coo.multiply(alpha_all).coalesce()
        debug_grad("reward_sim", reward_sim)

        # cost計算の勾配追跡
        reward_cost = edge.multiply(beta_all).to_sparse().coalesce()
        debug_grad("reward_cost", reward_cost)
        
        reward = reward_sim - reward_cost + reward_impact
        debug_grad("reward", reward)

        reward_loss = torch.sum(reward)
        debug_grad("reward_loss", reward_loss)

        optimizer.zero_grad()
        reward_loss.backward()
        optimizer.step()
        print("alpha,beta,gamma_grad",alpha_all.grad[10],beta_all.grad[10],gamma_all.grad[10])

        return alpha_all,beta_all,gamma_all




    #隣接行列を返す
    def state(self):

        one_hop_neighbar = self.edges#1hop
        two_hop_neighbar = remove_diagonal(torch.sparse.mm(self.edges,self.edges))#2hop

        return one_hop_neighbar, two_hop_neighbar, self.feature
