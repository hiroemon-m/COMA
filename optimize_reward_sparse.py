# Standard Library

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# First Party Library
import config
from init_real_data import init_real_data
import gc

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
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        return

    def optimize(self, t: int):

        feat = self.feats[t]
        edge = self.edges[t]
        self.optimizer.zero_grad()
        feat_dense = feat.to_dense()
        norm = torch.unsqueeze(torch.norm(feat_dense,p=2,dim=1),1)

        #norm = feat.norm(dim=1)[:, None] + 1e-8
        feat_norm = feat_dense.div(norm+1e-10)

        del feat_dense
        gc.collect()
        feat_norm = feat_norm.to_sparse()

        print("fn",feat_norm,edge)
        sim = torch.sparse.mm(feat_norm,feat_norm.transpose(0,1))
        


        #dot_product = torch.matmul(feat_norm, torch.t(feat_norm)).to(device)
        #sim = torch.mul(edge, dot_product)
        #sim = torch.mul(sim, self.model.alpha)
        #sim = torch.add(sim, 0.001)
        #costs = torch.mul(edge, self.model.beta)
        #costs = torch.add(costs, 0.001)
        #reward = torch.sub(sim, costs)
        #adj_feat = torch.sparse.mm(edge,feat_norm)
     
        #sim = torch.sparse.mm(adj_feat,feat_norm.transpose(0,1))
        sim = sim.multiply(edge)

        print("sim",sim,torch.sum(sim))
        print(sim.size())
        print(self.model.alpha.size())
        sim = torch.sparse.mm(sim, self.model.alpha).to_sparse()
        #print(sim)
        costs = torch.sparse.mm(edge,self.model.beta).to_sparse()
        #print("c",costs)
        reward = torch.sum(torch.sub(sim, costs))
        #print("BEtr",reward.sum())
   

       

        #if t > 0:
        #    trend = (torch.sum(self.feats[t-1],dim=0)>0).repeat(500,1)
        #    trend = torch.where(trend>0,1,0)
            #trend = torch.sum(self.feats[t-1],dim=0).repeat(500,1)
        #    trend =  (trend - self.feats[t])/len(self.feats[0][0])
            #trend =  (trend - self.feats[t])
    

        #    impact = (trend*self.model.gamma.view(500,-1)).sum()

    
                

        if t > 0:

            new_feature = torch.sparse.mm(self.edges[t],self.feats[t])
            new_feature = torch.sparse.mm(new_feature,new_feature.t())
            old_feature = torch.sparse.mm(self.edges[t-1], self.feats[t-1])
            old_feature = torch.sparse.mm(old_feature,old_feature.t())

            

            sub = torch.sub(old_feature,new_feature).abs()
            #print("nosb",torch.sum(old_feature - new_feature))
            sub = torch.sum(sub,dim=1,keepdim=True)
            #print("sub",torch.sum(sub))
            #print(new_feature)
            #print(old_feature)
            #print("s",sub,data.feature[0].size()[1])
            sub = sub/data.feature[0].size()[1]
            #print("scalesub",torch.sum(sub))
            #print(sub.size(),self.model.gamma.size())
            #print("tr",torch.sum(torch.sparse.mm(sub.t(),self.model.gamma)))
            reward += torch.sum(
                #self.model.gamma/torch.sqrt(torch.sum(torch.abs(new_feature - old_feature)**2))
                torch.sparse.mm(sub.t(),self.model.gamma)
                #self.model.gamma/(torch.sqrt(torch.abs(new_feature - old_feature)+1e-4)**2)
            )

        #    print(torch.abs(new_feature - old_feature)+1e-4)
        
        print("rs",reward)
  
        loss = -reward
        loss.backward()
        del loss
      
        self.optimizer.step()

       


    def export_param(self,data_type,data_name):
        #gamma/NIPS/
        with open("optimize/{}/{}/model.param.data.fast".format(data_type,data_name), "w") as f:
            max_alpha = 1.0
            max_beta = 1.0
            max_gamma = 1.0
            print("maxmax",max(self.model.alpha))
            print("maxmax",max(self.model.beta))
            print("maxmax",max(self.model.gamma))
            for i in range(self.size):
                f.write(
                    "{},{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                        self.model.gamma[i].item() / max_gamma,
                    )
                )


if __name__ == "__main__":
    start = time.perf_counter()
    data_name = "Twitter"

    data = init_real_data(data_name)
    data_size = data.adj[0].size()[0]

   


    alpha = torch.ones(data_size,1)
    beta = torch.ones(data_size,1)
    gamma = torch.ones(data_size,1)
    model = Model(alpha, beta, gamma)

    


    optimizer = Optimizer(data.adj, data.feature, model, data_size)
    data_type = "complete"

    for t in range(5):
        
        optimizer.optimize(t)
        optimizer.export_param(data_type,data_name)
    end = time.perf_counter()
    print((end-start)/60)