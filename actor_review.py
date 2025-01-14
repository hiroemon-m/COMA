import torch
import torch.nn as nn
import gc
import time
from torchviz import make_dot
import sys



def tanh_func(data,data_name="DBLP"):
    """スパーステンソルのtanh計算."""
    tanhx = torch.tanh(data.values())
    tanhx_indices = tanhx.nonzero(as_tuple=True)[0]
    filtered_indices = data.indices()[:, tanhx_indices]
    filtered_values = tanhx[tanhx_indices]

    if data_name == "Twitter":
        mask = filtered_indices[0] != filtered_indices[1]
        new_indices = filtered_indices[:, mask]
        new_values = filtered_values[mask]
        return torch.sparse_coo_tensor(new_indices, new_values, data.size()).coalesce()

    else:
        return torch.sparse_coo_tensor(filtered_indices, filtered_values, data.size()).coalesce()

def sigmoid_func(feat_data):
    """スパーステンソルのsigmoid計算."""
    feat_data = feat_data.coalesce()
    feat_sigmoid = torch.sigmoid(feat_data.values())
    return torch.sparse_coo_tensor(feat_data.indices(), feat_sigmoid, feat_data.size()).coalesce()

def calcu_l2(feat_data):
    """スパーステンソルのL2ノルムで正規化."""
    norm_indices = feat_data.indices()
    norm_values = feat_data.values()
    row_indices = norm_indices[0]
    squared_values = norm_values ** 2
    row_sums = torch.zeros(feat_data.size(0), device=feat_data.device)
    row_sums.index_add_(0, row_indices, squared_values)
    l2_norms = torch.sqrt(row_sums + 1e-10)
    normalized_values = norm_values / l2_norms[row_indices]
    return torch.sparse_coo_tensor(norm_indices, normalized_values, feat_data.size()).coalesce()

def sparse_hadamard_product(adj, similarity):
    """
    スパーステンソルのアダマール積を計算。共通インデックスのみを効率的に扱う。
    """
    # スパーステンソルを圧縮
    adj = adj.coalesce()
    similarity = similarity.coalesce()

    # 非ゼロ要素のインデックスと値を取得
    indices_a, values_a = adj.indices(), adj.values()
    indices_b, values_b = similarity.indices(), similarity.values()

    # (行, 列) を結合してユニークなキーとして扱う
    indices_a_flat = indices_a[0] * adj.size(1) + indices_a[1]
    indices_b_flat = indices_b[0] * similarity.size(1) + indices_b[1]

    # 共通インデックスを特定
    common_mask = torch.isin(indices_a_flat, indices_b_flat)

    # 共通インデックスに対応する値を取得
    common_indices = indices_a[:, common_mask]
    common_values_a = values_a[common_mask]

    # `indices_a_flat` と `indices_b_flat` の対応を見つけて、`values_b` を合わせる
    matched_b_indices = torch.searchsorted(indices_b_flat, indices_a_flat[common_mask])
    common_values_b = values_b[matched_b_indices]

    # アダマール積（要素ごとの積）を計算
    common_values = common_values_a * common_values_b

    # スパーステンソルとして返す
    result = torch.sparse_coo_tensor(common_indices, common_values, adj.size())
    result = result.coalesce()
    return result


def adj_sim(adj, feat):
    """隣接行列の類似度計算."""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    normalized_feat = calcu_l2(feat)
    similarity = torch.sparse.mm(normalized_feat, normalized_feat.t()).coalesce()
    neigh_similarity = sparse_hadamard_product(adj_sparse, similarity).coalesce()

    del similarity,normalized_feat,adj_sparse,feat_sparse,feat,adj
    gc.collect()

    return neigh_similarity






def remove_diagonal(data):
    """スパーステンソルの対角成分を削除."""
    filtered_indices = data.indices()
    mask = filtered_indices[0] != filtered_indices[1]
    new_indices = filtered_indices[:, mask]
    new_values = data.values()[mask]
    return torch.sparse_coo_tensor(new_indices, new_values, data.size()).coalesce()

def overwrite_exit(exit_prob, common_prob):
    """スパーステンソルの値を上書き."""
    indices_exit = exit_prob.indices()
    values_exit = exit_prob.values()
    indices_common = common_prob.indices()
    values_common = common_prob.values()

    matched_indices = torch.where(
        (indices_exit.T == indices_common.T.unsqueeze(1)).all(dim=2)
    )[1]
    updated_values = values_exit.clone()
    updated_values.index_copy_(0, matched_indices, values_common)

    return torch.sparse_coo_tensor(
        indices_exit, updated_values, exit_prob.size()
    ).coalesce()


class Actor(nn.Module):

    def __init__(self, T, e, r, w, persona, agent_num, temperature):
        super().__init__()
        size = (len(persona[0][0]),1)
        indices = torch.tensor([[0, 0, 0, 0, 0],[0, 1, 2, 3, 4]])  # 2次元に拡張
        print(size,T.detach().clone())
        self.T = nn.Parameter(T.clone().detach(),requires_grad=True)  # 密テンソル
        self.e = nn.Parameter(e.clone().detach(),requires_grad=True)  # 密テンソル
        self.r = nn.Parameter(r.clone().detach(),requires_grad=True)  # 密テンソル
        self.W = nn.Parameter(w.clone().detach(),requires_grad=True)  # 密テンソル
        self.temperature = temperature
        self.persona = persona
        self.agent_num = agent_num

        self.test_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)


    def _process_similarity(self, edges, attributes, weight, temp):
        """類似度と確率の計算."""
        
        adj_sim_matrix = adj_sim(edges, attributes)
        exp_input = adj_sim_matrix / temp

        exp_output = torch.exp(exp_input.values())
        return torch.sparse_coo_tensor(
            exp_input.indices(),
            exp_output * weight,
            exp_input.size()
        ).coalesce()

    def _combine_sparse_tensors(self, tensor_a, tensor_b):
        """スパーステンソルの結合."""
        indices = torch.cat([tensor_a.indices(), tensor_b.indices()], dim=1)
        values = torch.cat([tensor_a.values(), tensor_b.values()])
        return torch.sparse_coo_tensor(indices, values, tensor_a.size()).coalesce()

    def _create_two_hop_discconect(self,two_hop,one_hop):
        """2-hopの未接続ノードとのエッジ確率の計算."""
        indices_A = two_hop.indices()
        values_A = two_hop.values()
        size = two_hop.size()
        indices_B = one_hop.indices()
        values_B = one_hop.values()
      
        # A のインデックスをセットに変換
    
        indices_A_set = set(map(tuple, indices_A.T.tolist()))
        # B のインデックスをセットに変換
        indices_B_set = set(map(tuple, indices_B.T.tolist()))

        # A から B のインデックスを引く
        updated_indices_set = indices_A_set - indices_B_set

        # 更新後のインデックスをリストに変換
        #print("updated_indices",len(indices_A_set),len(indices_B_set))

        if len(updated_indices_set) == 0:
            updated_indices = torch.tensor([[0],[0]], dtype=torch.long) # 転置して [2, nnz] 形式にする
        else:
            updated_indices = torch.tensor(list(updated_indices_set), dtype=torch.long).T  # 転置して [2, nnz] 形式にする
        #print("updated_indices",updated_indices.size())
        # updated_values のサイズを updated_indices の列数に合わせる
        updated_values = torch.ones(updated_indices.size(1), dtype=torch.float32)
       
        # 新しいスパース隣接行列 C を作成
        create_edge = torch.sparse_coo_tensor(updated_indices, updated_values, size).coalesce()

        return create_edge

    def forward(self, attributes, edges, two_hop_neighbar, times, agent_num, sparse_size):

        #2hopの未接続ノードとのエッジ確率の計算

        for i in range(len(self.persona[0][0])):
            edges = edges.coalesce().clone().detach()
            attributes = attributes.coalesce().clone().detach()
            two_hop_neighbar = two_hop_neighbar.coalesce().clone().detach()
            # 属性値更新 - sparse.mmの結果に対して直接演算を行う
            neigh_feat_base = torch.sparse.mm(edges, attributes)
      
            # W[i]との乗算 - 密テンソルとして計算
            #neigh_feat = neigh_feat_base * self.W[i]
            neigh_feat = neigh_feat_base * self.W[i]

      
        
            # スケーリング計算を修正
            scaled_attributes = attributes * self.r[i]
            #scaled_attributes = attributes.multiply(self.r_sparse.values()[i])
     

            scaled_neigh_feat = neigh_feat * (1 - self.r[i])
            #scaled_neigh_feat = neigh_feat * (1 - self.r.values()[i])

            #scaled_neigh_feat = neigh_feat.multiply((1 - self.r_sparse.values()[i]))


            # 属性値の更新を結合 - 明示的な加算
            next_feat = scaled_attributes + scaled_neigh_feat
                # スパーステンソルの加算
            #next_feat = self._combine_sparse_tensors(scaled_attributes, scaled_neigh_feat)

            # 属性値の確率
            feat_sigmoid_prob = sigmoid_func(next_feat)

            # 2-hop以内の未接続ノードとのエッジ確率
            two_hop_edges = two_hop_neighbar.detach().clone()
            two_hop_disconnect_edges = self._create_two_hop_discconect(two_hop_edges,edges.coalesce())

            # 2-hopの未接続ノードとのエッジ確率の計算   
            #print("two_hop_disconnect_edges")
            connect_edge = self._process_similarity(two_hop_disconnect_edges, next_feat, self.e[i], self.T[i])
            create_edge_prob = tanh_func(connect_edge)

            # エッジ削除確率
            #print("edgedelete")
            one_hop_similality = adj_sim(edges, calcu_l2(next_feat))
            dissim_values = (1 - one_hop_similality.values())
            #print("1hop先のノード",edges._nnz())
            #print("one_hop_similality",one_hop_similality._nnz())
            dissim = torch.sparse_coo_tensor(
                one_hop_similality.indices(), dissim_values, one_hop_similality.size()
            ).coalesce()
            delete_edge = self._process_similarity(dissim, next_feat, self.e[i], self.T[i])

            delete_edge_prob = tanh_func(delete_edge)
 
            #加算
            exit_edge_prob = create_edge_prob + delete_edge_prob
            #print("exit_edge_prob",exit_edge_prob._nnz())
            #print("create_edge_prob",create_edge_prob._nnz())
            #print("delete_edge_prob",delete_edge_prob._nnz())
            #dot = make_dot(exit_edge_prob,params=dict(list(self.named_parameters())))
            #dot = make_dot(connect_edge,params=dict(list(self.named_parameters())))

            # グラフを表示
            #dot.format = "svg"
            #dot.render("graph_ab", format="png", cleanup=True)
            #break
        
            

            # エッジ確率の結合
            if i == 0:
                edges_prob = self.persona[times][:, i] * exit_edge_prob
            else:
                persona_weighted = self.persona[times][:, i] * exit_edge_prob
                edges_prob = self._combine_sparse_tensors(edges_prob, persona_weighted)

            # 属性確率の結合
            if i == 0:
                attr_prob = feat_sigmoid_prob * self.persona[times][:, i].view(-1, 1)
            else:
                persona_weighted_attr = feat_sigmoid_prob * self.persona[times][:, i].view(-1, 1)
                attr_prob = self._combine_sparse_tensors(attr_prob, persona_weighted_attr)
        
        
        return edges_prob, attr_prob, feat_sigmoid_prob, next_feat, scaled_attributes, scaled_neigh_feat

    def train(self, attributes, edges, times, agent_num, sparse_size):

        #2hop以内の隣接行列
        edge = torch.sparse.mm(edges,edges)

        #2hop先の隣接行列の対角成分を取り除く
        two_hop_neighbar = remove_diagonal(edge)


        edges_prob, attr_prob,feat_sigmoid_prob,next_feat,scaled_attributes,scaled_neigh_feat = self.forward(attributes, edges, two_hop_neighbar, times, agent_num, sparse_size)
        #dot = make_dot(edges_prob,params=dict(list(self.named_parameters())))
        #dot.format = "svg"
        # グラフを表示
        #dot.render("graph", format="png", cleanup=True)

        return edges_prob
        
    def get_action(self, attributes, edges, two_hop_neighbar, times, agent_num, sparse_size):
        with torch.no_grad():
            edges_prob, attr_prob ,_,_,_,_ = self.forward(attributes, edges, two_hop_neighbar, times, agent_num, sparse_size)

            # 属性アクション
            attr_prob_values = attr_prob.values()
            attr_value = torch.bernoulli(torch.sigmoid(attr_prob_values))
            non_zero_indices = attr_value.nonzero(as_tuple=True)[0]
            filtered_indices = attr_prob.indices()[:, non_zero_indices]
            filtered_values = attr_prob_values[non_zero_indices]
            attr_action = torch.sparse_coo_tensor(filtered_indices, filtered_values, attr_prob.size())

            # エッジアクション
            edge_value = (edges_prob.values() >= 0.5).float()
            non_zero_indices = edge_value.nonzero(as_tuple=True)[0]
            filtered_indices = edges_prob.indices()[:, non_zero_indices]
            filtered_values = edge_value[non_zero_indices]
            edge_action = torch.sparse_coo_tensor(filtered_indices, filtered_values, edges_prob.size()).coalesce()

        return edges_prob, edge_action, attr_action

    def pred(self, attributes, edges, two_hop_neighbar, times, agent_num, sparse_size):
        with torch.no_grad():
            edges_prob, attr_prob ,_,_,_,_ = self.forward(attributes, edges, two_hop_neighbar, times, agent_num, sparse_size)

            # 属性アクション
            attr_prob_values = attr_prob.values()
            attr_value = torch.bernoulli(torch.sigmoid(attr_prob_values))
            non_zero_indices = attr_value.nonzero(as_tuple=True)[0]
            filtered_indices = attr_prob.indices()[:, non_zero_indices]
            filtered_values = attr_prob_values[non_zero_indices]
            attr_action = torch.sparse_coo_tensor(filtered_indices, filtered_values, attr_prob.size())

            # エッジアクション
            edge_value = (edges_prob.values() >= 0.5).float()
            non_zero_indices = edge_value.nonzero(as_tuple=True)[0]
            filtered_indices = edges_prob.indices()[:, non_zero_indices]
            filtered_values = edge_value[non_zero_indices]
            edge_action = torch.sparse_coo_tensor(filtered_indices, filtered_values, edges_prob.size()).coalesce()

        return edge_action, edges_prob, attr_prob, attr_action

    


