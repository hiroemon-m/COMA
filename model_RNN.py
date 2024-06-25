import numpy as np
import copy
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

eps = sys.float_info.min


def inv_sigmoid(x):
    return torch.log((1+x)/(1-x))
def sp_sigmoid(x):
    return 2*torch.sigmoid(x) - 1.

def loss_function_deterministic(model, preds_edge, labels_edge, preds_negedge, labels_negedge, predicted_U, predicted_V, predicted_X, preds_attr, labels_attr, preds_negattr, labels_negattr):
    cost_edge = 0
    cost_attr = 0
    cost_reg  = 0
    for t in range(model.T_train):
        cur_cost_edge = 0
        cur_cost_attr = 0
        cur_cost_reg = 0
        cur_cost_edge += F.binary_cross_entropy(preds_edge[t], labels_edge[t], reduction='sum')
        cur_cost_attr += F.binary_cross_entropy(preds_attr[t], labels_attr[t], reduction='sum')
        cur_cost_edge += F.binary_cross_entropy(preds_negedge[t], labels_negedge[t], reduction='sum')
        cur_cost_attr += F.binary_cross_entropy(preds_negattr[t], labels_negattr[t], reduction='sum')
        if False:
            print(cost_edge)
            print(cost_attr)
            print(cost_reg)
            print(cost_pred)
        # dec_t = (t+1)/(model.T_train+1) # linear decaying
        dec_t = 1.
        cost_edge += dec_t * cur_cost_edge
        cost_attr += dec_t * cur_cost_attr
        cost_reg  += dec_t * cur_cost_reg

    return cost_edge + cost_attr + cost_reg
    # return cost_edge + cost_reg
    # return cost_attr + cost_reg

class predSN_RNN(Module):
    def __init__(self, T_train, N_nodes, N_words, dim_edge, dim_interest, dropout=0., svd=None, device="1"):
        super(predSN_RNN, self).__init__()
        self.T_train = T_train
        self.device = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
        self.N_nodes = N_nodes
        self.N_words = N_words
        self.dim_edge = dim_edge
        self.dim_interest = dim_interest

        self.define_weights()
        self.initialize_weights()
        if svd is not None:
            self.initialize_svd(svd)
        
        self.dc_edge = InnerProductDiEdgeDecoder(dropout, dim_interest=dim_interest, dim_edge=dim_edge, act=lambda x: x)
        self.dc_attr = InnerProductAttrDecoder  (dropout, dim_interest=dim_interest, act=lambda x: x)
        self.pred_V = EdgeEmbeddingPredict(dropout, act=lambda x: x, N_nodes1=self.N_nodes, N_nodes2=self.N_words, dim=dim_interest)
        self.pred_U = EdgeEmbeddingPredict(dropout, act=lambda x: x, N_nodes1=self.N_nodes, N_nodes2=self.N_words, dim=dim_edge)
        self.pred_X = AttrEmbeddingPredict(dropout, act=lambda x: x, N_nodes1=self.N_words, N_nodes2=self.N_nodes, dim=dim_interest)
    
    def load_train(self, adj, adj_neg, feature, feature_neg):
        self.adj = adj
        self.adj_neg = adj_neg
        self.feature = feature
        self.feature_neg = feature_neg

    def define_weights(self):
        if torch.cuda.is_available():
            self.U = Parameter(torch.FloatTensor(self.T_train, self.N_nodes, self.dim_edge*2    ).cuda())
            self.V = Parameter(torch.FloatTensor(self.T_train, self.N_nodes, self.dim_interest*2).cuda())
            self.X = Parameter(torch.FloatTensor(self.T_train, self.N_words, self.dim_interest  ).cuda())
        else:
            self.U = Parameter(torch.FloatTensor(self.T_train, self.N_nodes, self.dim_edge*2    ))
            self.V = Parameter(torch.FloatTensor(self.T_train, self.N_nodes, self.dim_interest*2))
            self.X = Parameter(torch.FloatTensor(self.T_train, self.N_words, self.dim_interest  ))

    def initialize_weights(self):
        '''
        Initializing the embeddings.
        '''
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.X)
    
    def initialize_svd(self, svd):
        '''
        Initialize model via result of SVD.
        '''
        self.Vin_svd = svd[0]
        self.Vout_svd = svd[1]
        self.X_svd = svd[2]
        self.s = svd[3]

    def forward(self):
        reconst_edge = []
        reconst_attr = []
        reconst_negedge = []
        reconst_negattr = []
        
        U_future = []
        V_future = []
        X_future = []
        
        V = self.V[0]
        U = self.U[0]
        X = self.X[0]

        for t in range(self.T_train):
            for mode in ['pos', 'neg']:
                if mode=='pos':
                    ind_G = self.adj[t]._indices()
                    val_G = self.adj[t]._values()
                    ind_A = self.feature[t]._indices()
                    val_A = self.feature[t]._values()
                else:
                    ind_G = self.adj_neg[t]._indices()
                    val_G = self.adj_neg[t]._values()
                    ind_A = self.feature_neg[t]._indices()
                    val_A = self.feature_neg[t]._values()
                ind_Gin = ind_G[0,:]
                ind_Gout = ind_G[1,:]
                ind_Anode = ind_A[0,:]
                ind_Aattr = ind_A[1,:]
                Uout = U[ind_Gout, :self.dim_edge]
                Uin  = U[ind_Gin,  self.dim_edge:]
                Vout = V[ind_Gout, :self.dim_interest]
                Vin  = V[ind_Gin,  self.dim_interest:]
                _reconst_edge = self.dc_edge(Vout, Vin, Uout, Uin)
                _reconst_attr = self.dc_attr(self.V[t,ind_Anode, :self.dim_interest], self.X[t,ind_Aattr])
                if mode=='pos':
                    reconst_edge.append(_reconst_edge)
                    reconst_attr.append(_reconst_attr)
                else:
                    reconst_negedge.append(_reconst_edge)
                    reconst_negattr.append(_reconst_attr)

            if t != self.T_train-1:
                U = self.pred_U(U, self.adj[t], X, self.feature[t])
                V = self.pred_V(V, self.adj[t], X, self.feature[t])
                X = self.pred_X(X, V, self.feature[t].t())
                
        return reconst_edge, reconst_negedge, reconst_attr, reconst_negattr, U_future, V_future, X_future
    
    def forecast(self, test_adj, test_adj_neg, test_feature, test_feature_neg):
        future = len(test_adj)
        
        reconst_edge = []
        reconst_attr = []
        reconst_negedge = []
        reconst_negattr = []

        V_ = []
        U_ = []
        X_ = []

        V_.append(self.pred_V(self.V[0], self.adj[0], self.X[0], self.feature[0]))
        U_.append(self.pred_U(self.U[0], self.adj[0], self.X[0], self.feature[0]))
        X_.append(self.pred_X(self.X[0], self.V[0], self.feature[0].t()))
        for t in range(1,self.T_train):
            V_.append(self.pred_V(V_[-1], self.adj[t], X_[-1], self.feature[t]))
            U_.append(self.pred_U(U_[-1], self.adj[t], X_[-1], self.feature[t]))
            X_.append(self.pred_X(X_[-1], V_[-1], self.feature[t].t()))

        for t in range(future):
            for mode in ['pos', 'neg']:
                if mode=='pos':
                    ind_G = test_adj[t]._indices()
                    val_G = test_adj[t]._values()
                    ind_A = test_feature[t]._indices()
                    val_A = test_feature[t]._values()
                else:
                    ind_G = test_adj_neg[t]._indices()
                    val_G = test_adj_neg[t]._values()
                    ind_A = test_feature_neg[t]._indices()
                    val_A = test_feature_neg[t]._values()
                ind_Gin = ind_G[0,:]
                ind_Gout = ind_G[1,:]
                ind_Anode = ind_A[0,:]
                ind_Aattr = ind_A[1,:]
                Uout = U_[-1][ind_Gout, :self.dim_edge]
                Uin  = U_[-1][ind_Gin,  self.dim_edge:]
                Vout = V_[-1][ind_Gout, :self.dim_interest]
                Vin  = V_[-1][ind_Gin,  self.dim_interest:]
                _reconst_edge = self.dc_edge(Vout, Vin, Uout, Uin)
                _reconst_attr = self.dc_attr(V_[-1][ind_Anode, :self.dim_interest], X_[-1][ind_Aattr])
                if mode=='pos':
                    reconst_edge.append(_reconst_edge)
                    reconst_attr.append(_reconst_attr)
                else:
                    reconst_negedge.append(_reconst_edge)
                    reconst_negattr.append(_reconst_attr)
            # self.adj.append(test_adj[t])
            # self.feature.append(test_feature[t])

            V_.append(self.pred_V(V_[-1], self.adj[-1], X_[-1], self.feature[-1]))
            U_.append(self.pred_U(U_[-1], self.adj[-1], X_[-1], self.feature[-1]))
            X_.append(self.pred_X(X_[-1], V_[-1], self.feature[-1].t()))
        return reconst_edge, reconst_negedge, reconst_attr, reconst_negattr

class InnerProductDiEdgeDecoder(Module):
    """Decoder for using inner product for edge prediction."""

    def __init__(self, dropout, dim_interest, dim_edge, act=torch.sigmoid):
        super(InnerProductDiEdgeDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        if torch.cuda.is_available():
            self.W = Parameter(torch.eye(dim_interest).cuda())
            self.W2 = Parameter(torch.eye(dim_edge).cuda())
            self.b = Parameter(torch.FloatTensor([0.]).cuda())
        else:
            self.W = Parameter(torch.eye(dim_interest))
            self.W2 = Parameter(torch.eye(dim_edge))
            self.b = Parameter(torch.FloatTensor([0.]))
        self.initialize_weights()

    def initialize_weights(self):
        # torch.nn.init.xavier_uniform_(self.W)
        pass
    
    def forward(self, v_out, v_in, u_out, u_in):
        # adj = torch.sigmoid(torch.sum(u_out * u_in, 1))
        adj = torch.sigmoid(torch.sum(v_out * v_in, 1))
        # adj = torch.sigmoid(torch.sum(torch.mm(v_out, self.W) * v_in, 1) + torch.sum(torch.mm(u_out, self.W2) * u_in, 1) + self.b)

        return adj
    
class InnerProductAttrDecoder(Module):
    """Decoder for using inner product for attribute prediction."""

    def __init__(self, dropout, dim_interest, act=torch.sigmoid):
        super(InnerProductAttrDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        if torch.cuda.is_available():
            # self.W = Parameter(torch.zeros(dim_interest * 2, dim_interest))
            self.W = Parameter(torch.zeros(dim_interest, dim_interest).cuda())
            self.b = Parameter(torch.FloatTensor([0.]).cuda())
        else:
            # self.W = Parameter(torch.zeros(dim_interest * 2, dim_interest))
            self.W = Parameter(torch.zeros(dim_interest, dim_interest))
            self.b = Parameter(torch.FloatTensor([0.]))
        self.initialize_weights()

    def initialize_weights(self):
        # torch.nn.init.xavier_uniform_(self.W)
        pass
    
    def forward(self, v, x):
        feature = torch.sigmoid(torch.sum(torch.mm(v, self.W) * x, 1) + self.b)
        return feature


class EdgeEmbeddingPredict(Module):
    """Edge embedding prediction."""

    def __init__(self, dropout, N_nodes1, N_nodes2, dim, act=torch.sigmoid):
        super(EdgeEmbeddingPredict, self).__init__()
        self.dropout = dropout
        self.act = act
        self.N_nodes1 = N_nodes1
        self.N_nodes2 = N_nodes2
        if torch.cuda.is_available():
            self.m  = Parameter(torch.FloatTensor(N_nodes1,1).cuda())
            self.w  = Parameter(torch.FloatTensor(N_nodes1,1).cuda())
            self.W1 = Parameter(torch.eye(dim*2).cuda())
            self.W2 = Parameter(torch.eye(dim*2).cuda())
            self.W3 = Parameter(torch.zeros(N_nodes1, dim*2).cuda())
        else:
            self.m  = Parameter(torch.FloatTensor(N_nodes1,1))
            self.w  = Parameter(torch.FloatTensor(N_nodes1,1))
            self.W1 = Parameter(torch.eye(dim*2))
            self.W2 = Parameter(torch.eye(dim*2))
            self.W3 = Parameter(torch.zeros(N_nodes1, dim*2))
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.W1)
        # torch.nn.init.xavier_uniform_(self.W2)
        # torch.nn.init.xavier_uniform_(self.W3)
        nn.init.constant_(self.W3,  0.000)
        # nn.init.constant_(self.m,  0.)
        # nn.init.constant_(self.w,  0.)
        nn.init.constant_(self.m,  0.1)
        nn.init.constant_(self.w,  0.1)

    def forward(self, v1, adj_1, v2, adj_2):
        # v_ = torch.sigmoid(torch.mm(v1, self.W1) + self.W3)
        v_ = torch.mm(v1, self.W1) + self.W3

        # v_ = torch.mm(v1, self.W1) + F.relu(self.m) * torch.mm(torch.spmm(adj_1, F.relu(self.w) * v1) / (torch.spmm(adj_1, F.relu(self.w))+10e-15), self.W2) + self.W3

        return v_

class AttrEmbeddingPredict(Module):
    """Edge embedding prediction."""

    def __init__(self, dropout, N_nodes1, N_nodes2, dim, act=torch.sigmoid):
        super(AttrEmbeddingPredict, self).__init__()
        self.dropout = dropout
        self.act = act
        self.N_nodes1 = N_nodes1
        self.N_nodes2 = N_nodes2
        if torch.cuda.is_available():
            self.m  = Parameter(torch.FloatTensor(N_nodes1,1).cuda())
            self.w  = Parameter(torch.FloatTensor(N_nodes2,1).cuda())
            self.W1 = Parameter(torch.eye(dim).cuda())
            self.W2 = Parameter(torch.cat([torch.eye(dim),torch.eye(dim)],dim=0).cuda())
            self.W3 = Parameter(torch.zeros(N_nodes1, dim).cuda())
        else:
            self.m  = Parameter(torch.FloatTensor(N_nodes1,1))
            self.w  = Parameter(torch.FloatTensor(N_nodes2,1))
            self.W1 = Parameter(torch.eye(dim))
            self.W2 = Parameter(torch.cat([torch.eye(dim),torch.eye(dim)],dim=0))
            self.W3 = Parameter(torch.zeros(N_nodes1, dim))
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.W1)
        # torch.nn.init.xavier_uniform_(self.W2)
        # torch.nn.init.xavier_uniform_(self.W3)
        nn.init.constant_(self.W3,  0.000)
        nn.init.constant_(self.m, 0.1)
        nn.init.constant_(self.w, 0.1)


    def forward(self, v1, v2, adj):
        # v_ = torch.sigmoid(torch.mm(v1, self.W1) + self.W3)

        v_ = torch.mm(v1, self.W1) + self.W3

        # v_ = torch.mm(v1, self.W1) + F.relu(self.m) * torch.mm(torch.spmm(adj, F.relu(self.w) * v2) / (torch.spmm(adj, F.relu(self.w))+10e-15), self.W2) + self.W3

        return v_