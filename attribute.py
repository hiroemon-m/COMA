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


class Actor_attribute(nn.Module):
    def __init__(self, r, w,persona) -> None:
        super().__init__()

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
        attributes = attributes.div(norm)
        #print(self.persona[:,1])
        #print(self.persona)
        edges = (edges > 0).float().to(device)
        probability = 0
        for i in range(len(self.persona[0])):
            #隣接ノードと自分の特徴量を集約する
            tmp_tensor = self.W[i] * torch.matmul(edges, attributes)
            r = self.r[i]
            r = r + 1e-8
            feat = r * attributes + tmp_tensor * (1 - r)
 

        return feat



    def predict(
        self,attributes, edges
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return self.forward(attributes, edges)