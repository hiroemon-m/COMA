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

class Interest(IntEnum):
    RED = 2
    BLUE = 1


class Model(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()

        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        return


class Optimizer:
    def __init__(self, edges, feats, model: Model, size: int):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        return

    def optimize(self, t: int):
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device)
        self.optimizer.zero_grad()
        dot_product = torch.matmul(feat, torch.t(feat)).to(device)
        sim = torch.mul(edge, dot_product)
        sim = torch.mul(sim, self.model.alpha)
        sim = torch.add(sim, 0.001)

        costs = torch.mul(edge, self.model.beta)
        #print(costs[n[0],n[1]])
        #print(t,edge[n[0],n[1]])
        costs = torch.add(costs, 0.001)
    
         

        reward = torch.sub(sim, costs)
        loss = -reward.sum()

        loss.backward()
        print(loss)
        del loss
        self.optimizer.step()
        


    def export_param(self,skiptime,p,k):

        with open("experiment_data/NIPS/incomplete/t={}/edge/percent={}/attempt={}/model.param.data.fast".format(skiptime,p,k), "w") as f:
            max_alpha = 1.0
            max_beta = 1.0

            for i in range(self.size):
                f.write(
                    "{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                    )
                )


if __name__ == "__main__":
    for p in [3,5,15,30,50,75]:

        for skiptime in [4]:
            for k in range(0,30):


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
                model = Model(alpha, beta)


                persona = 5
                    #あるノードにi関する情報を取り除く
                    #list[tensor]のキモい構造なので
                
                #エッジ
                #エッジの接続リスト
                edgeindex=data.adj[skiptime][:,:].nonzero(as_tuple=False)

                #エッジの本数
                edge_num = edgeindex.size()[0]
        
                #パーセントに応じて取り除く数を決める
                skipnum = int(edge_num*(p/100))
        

                skipindex = random.sample(range(0, edge_num), skipnum)
                for i in skipindex:
                    n = edgeindex[i]
        
                    #print(data.adj[skiptime][n.tolist()[0],n.tolist()[1]])
                    data.adj[skiptime][n.tolist()[0],n.tolist()[1]]=0
                    #print(data.adj[skiptime][n.tolist()[0],n.tolist()[1]])
                    
                    
            
             
                with open("experiment_data/NIPS/incomplete/t={}/edge/percent={}/attempt={}/delete_index".format(skiptime,p,k), "w") as f:
                    for i in skipindex:
                        f.write(
                                  "{}\n".format(str(i))
                                )
                #属性値消す
                #skipnum = int(data_size*(p/100))
                #skipindex = random.sample(range(0, data_size), skipnum)
                #for i in skipindex:

                    #print(data.feature[4][i][:])
                #    data.feature[skiptime][i][:] = 0
                    #print(data.adj[skiptime][n.tolist()[0],n.tolist()[1]])
                    #data.adj[skiptime][n.tolist()[0],n.tolist()[1]]=0
                    #print(data.adj[skiptime][n.tolist()[0],n.tolist()[1]])
                    #print(data.feature[4][i][:])
                    #print(1111)
                    
            
                    #data.feature[4][r][:] = 0
                #with open("experiment_data/NIPS/incomplete/t={}/edge/percent={}/attempt={}/delete_index".format(skiptime,p,k), "w") as f:
                #    for i in skipindex:
                #        f.write(
                #                "{}\n".format(str(i))
                #        )

                #print(data.adj[4][:,:].nonzero(as_tuple=False).size()) 
                optimizer = Optimizer(data.adj, data.feature, model, data_size)
                for t in range(5):
                    optimizer.optimize(t)
                optimizer.export_param(skiptime,p,k)