# Third Party Library
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch_geometric

class TimeSeriesGCN(nn.Module):
    def __init__(self, node_attr_dim, batch_size,hidden_dim=128, num_nodes=32):
        super(TimeSeriesGCN, self).__init__()
        self.conv1 = GCNConv(node_attr_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.batch_size = batch_size
        self.num_nodes = num_nodes

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = torch.reshape(x, (self.batch_size, self.num_nodes, self.num_nodes))
        return x

    def predict(self, edges, feat):
        batch_size = self.batch_size
        num_nodes = self.num_nodes

        # グラフデータをPyTorch Geometricの形式に変換
        graphs = []
        for i in range(batch_size):
            edge_index = (edges[i] == 1).nonzero(as_tuple=False).t()
            x = feat[i]
            data = Data(x=x, edge_index=edge_index)
            data.batch = torch.full((num_nodes,), i, dtype=torch.long)
            graphs.append(data)

        # バッチ化
        batch = torch_geometric.data.Batch.from_data_list(graphs)

        # フォワードパス
        output = self(batch.x, batch.edge_index, batch.batch)
        return output
