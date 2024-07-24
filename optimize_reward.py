# Standard Library
import random
from enum import IntEnum

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# First Party Library
import config
from init_real_data import init_real_data

device = config.select_device


class Model(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()

        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)
        return


class Optimizer:
    def __init__(self, edges, feats, model: Model, size: int):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)

        return

    def optimize(self, t: int):
        impact = 0
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device) 
        self.optimizer.zero_grad()
        norm = feat.norm(dim=1)[:, None] + 1e-8
        feat_norm = feat.div(norm)
        dot_product = torch.matmul(feat_norm, torch.t(feat_norm)).to(device)

        sim = torch.mul(edge, dot_product)
        sim = torch.mul(sim, self.model.alpha)
        sim = torch.add(sim, 0.001)

        costs = torch.mul(edge, self.model.beta)
        costs = torch.add(costs, 0.001)
        reward = torch.sub(sim, costs)

        #if t > 0:
        #    trend = (torch.sum(self.feats[t-1],dim=0)>0).repeat(500,1)
        #    trend = torch.where(trend>0,1,0)
            #trend = torch.sum(self.feats[t-1],dim=0).repeat(500,1)
        #    trend =  (trend - self.feats[t])/len(self.feats[0][0])
            #trend =  (trend - self.feats[t])
    

        #    impact = (trend*self.model.gamma.view(500,-1)).sum()

    
                

        if t > 0:
            print(self.edges[t])
            new_feature = torch.matmul(self.edges[t],self.feats[t])
            new_feature = torch.matmul(new_feature,torch.t(new_feature))
            old_feature = torch.matmul(self.edges[t-1], self.feats[t-1])
            old_feature = torch.matmul(old_feature,torch.t(old_feature))

            reward += (
                #self.model.gamma/torch.sqrt(torch.sum(torch.abs(new_feature - old_feature)**2))
                self.model.gamma*(torch.abs(new_feature - old_feature)+1e-4)/len(new_feature[0])
                #self.model.gamma/(torch.sqrt(torch.abs(new_feature - old_feature)+1e-4)**2)

            )
        #    print(torch.abs(new_feature - old_feature)+1e-4)

  
        loss = -reward.sum()
        loss.backward()
        del loss
        self.optimizer.step()

       


    def export_param(self,data_type,del_type,skiptime,p,k):
        #gamma/NIPS/
        with open("optimize/{}/DBLP/t={}/{}/percent={}/attempt={}/model_param".format(data_type,skiptime,del_type,p,k), "w") as f:
            max_alpha = 1.0
            max_beta = 1.0
            max_gamma = 1.0

            for i in range(self.size):
                f.write(
                    "{},{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                        self.model.gamma[i].item() / max_gamma,
                    )
                )


if __name__ == "__main__":
    for p in [5,15,30,50,75]:
        for skiptime in [4]:
            for k in range(10):
                for del_type in ["edge","attr"]:
                    data = init_real_data()
                    data_size = len(data.adj[0])

                    alpha = torch.from_numpy(
                        np.array(
                            [1.0 for i in range(data_size)],
                            dtype=np.float32,
                        ),
                    ).to(device)

                    beta = torch.from_numpy(
                        np.array(
                            [1.0 for i in range(data_size)],
                            dtype=np.float32,
                        ),
                    ).to(device)

                    gamma = torch.from_numpy(
                        np.array(
                            [1.0 for i in range(data_size)],
                            dtype=np.float32,
                        ),
                    ).to(device)
                    model = Model(alpha, beta, gamma)

                    if del_type == "edge":
                
                        #エッジの接続リスト
                        edgeindex=data.adj[skiptime][:,:].nonzero(as_tuple=False)
                        #エッジの本数
                        edge_num = edgeindex.size()[0]
                        #print("fn",edge_num)
                        #パーセントに応じて取り除く数を決める
                        skipnum = int(edge_num*(p/100))
                        #エッジの欠損
                        skipindex = random.sample(range(0, edge_num), skipnum)
                        for i in skipindex:
                            n = edgeindex[i]
                
                            data.adj[skiptime][n.tolist()[0],n.tolist()[1]]=0
                            #print("Edge",data.adj[skiptime])

                    else:#属性値消す
                        featureindex=data.feature[skiptime][:,:].nonzero(as_tuple=False)
                        feature_num = featureindex.size()[0]
                        #print("fn",feature_num)
                        skipnum = int(feature_num*(p/100))
                        skipindex = random.sample(range(0, feature_num), skipnum)
                        for i in skipindex:
                            n = featureindex[i]
                            #print("bf",data.feature[skiptime].sum())
                            data.feature[skiptime][n.tolist()[0],n.tolist()[1]]=0
                            #print(data.feature[skiptime].sum())
                          
                        
        
        
                
                    with open("optimize/imcomplete/DBLP/t={}/{}/percent={}/attempt={}/delete_index".format(skiptime,del_type,p,k), "w") as f:
                        for i in skipindex:
                            f.write(
                                    "{}\n".format(str(i))
                                    )
            
    

                    optimizer = Optimizer(data.adj, data.feature, model, data_size)
                    data_type = "imcomplete"
                    for t in range(5):
                        optimizer.optimize(t)
                        optimizer.export_param(data_type,del_type,skiptime,p,k)