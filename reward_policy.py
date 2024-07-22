# Standard Library


# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

# First Party Library
import config

device = config.select_device



class Model(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)

        return
        
class Optimizer:
    def __init__(self, memory, persona, model: Model):
        self.edges = memory.probs
        self.feats = memory.feat_probs
        self.old_edges = memory.next_edges
        self.old_feats = memory.next_features
        self.persona = persona
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

    
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
        y = self.gumbel_softmax_sample(logits, 0.05)

        if hard:
            # ハードサンプル：one-hotにするが、勾配はソフトサンプルに基づく
            shape = y.size()
            _, max_idx = y.max(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y).scatter_(-1, max_idx, 1.0)
            y = (y_hard - y).detach() + y
    
        return y


    def optimize(self, time: int):

        reward = 0
        feat_prob = torch.empty(len(self.feats[0][0]),len(self.feats[0][0][0]),2)
        edge_prob = torch.empty(len(self.edges[0][0]),len(self.edges[0][0][0]),2)

        persona_edge = self.edges[time]
        persona_feat = self.feats[time]
        persona = torch.unsqueeze(torch.transpose(self.persona[time],1,0),dim=-1)
        node_edge = persona_edge*persona
        node_feat = persona_feat*persona

        #edge,featのペルソナの方向に足したもの 5x32x32 -> 32 x 32
        edge_tanh = torch.clamp(torch.sum(node_edge,dim=0),min=0,max=1)
        feat_tanh = torch.clamp(torch.sum(node_feat,dim=0),min=0,max=1)


        #feat_prob[:,:,0] = 10 - (feat_tanh * 10)
        #feat_prob[:,:,1] = feat_tanh * 10
        #feat_action= self.gumbel_softmax(feat_prob,hard=True)
        #feat_action = feat_action[:,:,1]

        #edge_prob[:,:,0] = 10 - (edge_tanh*10)
        #edge_prob[:,:,1] = edge_tanh*10
        #edge_action= self.gumbel_softmax(edge_prob)[:,:,1]
        edge_action = edge_tanh
        feat_action = feat_tanh
        #sim
        norm = feat_action.norm(dim=1)[:, None] + 1e-8
        feat_norm = feat_action.div(norm)
        dot_product = torch.matmul(feat_norm, torch.t(feat_norm)).to(device)
        sim = torch.mul(edge_action, dot_product)
        sim = torch.mul(sim, self.model.alpha)
        sim = torch.add(sim, 0.001)
        #cost
        costs = torch.mul(edge_action, self.model.beta)
        costs = torch.add(costs, 0.001)
        reward = torch.sub(sim, costs)   
        print("rewaed",reward.size())
        #impact
        if time > 0:
            old_edge = self.old_edges[time]
            old_feat = self.old_feats[time]
            persona = torch.unsqueeze(torch.transpose(self.persona[time],1,0),dim=-1)
            old_edge = old_edge * persona
            old_feat = old_feat * persona

            new_feature = torch.matmul(node_edge,node_feat)
            old_feature = torch.matmul(self.old_edges[time], self.old_feats[time])

            reward += torch.sum((
                torch.matmul(self.model.gamma.view(1,-1).to(torch.float32),(torch.abs(new_feature - old_feature)+1e-4)))/len(new_feature[0][0])
            ).to(torch.float32)  
        #reward_clip=[-1,1]
        #reward = torch.clamp(reward,min=reward_clip[0],max=reward_clip[1])
        print("sum11",reward.sum())
        self.optimizer.zero_grad()
        loss = -reward.sum()
        loss.backward(retain_graph=True)
        self.optimizer.step()



