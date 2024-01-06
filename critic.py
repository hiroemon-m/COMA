# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, action_dim):
        super(Critic, self).__init__()

        input_dim = input_size
        action_dim = 500

        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, action_dim)

    def forward(self, x):
        
        #x = torch.nan_to_num(x)
        print("n2ndone")
        x = F.relu(self.fc1(x))
        print("fc1done")
        #x = torch.nan_to_num(x)
        x = F.relu(self.fc2(x))
        #x = torch.nan_to_num(x)
        x = self.fc3(x)
        #x_fixed = torch.nan_to_num(x)
        return x