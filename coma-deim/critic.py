# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv


class Critic(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(Critic, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, edge,feature):
        x = feature
        edge_index = edge

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x