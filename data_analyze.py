from init_real_data import init_real_data
import torch
from sklearn.preprocessing import MinMaxScaler,Normalizer

LEARNED_TIME = 4
GENERATE_TIME = 5
TOTAL_TIME = 10
load_data = init_real_data()


for time in range(TOTAL_TIME):

    edge = load_data.adj[time]
    attr = load_data.feature[time]

    print("-----------------{}----------------".format(time))
    print("属性値の総数")
    print(torch.sum(attr))

    print(torch.sum(attr,dim=1))

    if time >= 1:
        edge_before = load_data.adj[time-1]
        print(attr.size())
        attr_before = load_data.feature[time-1]
        print("一つ前の時刻と同じ属性値")
        sa = (attr>0.0)&(attr_before>0.0)
        print(torch.sum(sa,dim=1))
        old = 0
        for k in range(time):
            print(k)
            old = old + load_data.feature[time]
        print("kako",torch.sum((old>0)&(attr>0),dim=1))

  
        print("一つ前の時刻から０属性値")
        print(torch.sum(torch.where((torch.where(attr_before>0.0,1.0,0.0)-torch.where(attr>0.0,1.0,0.0))>0.0,1.0,0.0),dim=1))
        print("一つ前の時刻の隣接ノード同じ属性値")
        adj = torch.matmul(edge_before,attr_before)
        aa = (adj>0.0)&(attr>0.0)
        print(torch.sum(aa,dim=1))
        #print("一つ前の時刻の隣接でないノード同じ属性値")
        un_adj = torch.matmul(1-edge_before,attr_before)
        #print(torch.sum((un_adj>0.0)&(attr>0.0),dim=1))
        #print("一つ前の時刻の隣接でないノードと隣接ノードの同じ属性値")
        #print(torch.sum((un_adj>0.0)&(adj>0.0),dim=1))
        #隣接と隣接でないどちらかでいい？という説の否定。多すぎる少なすぎることが多い

        #仮説　一つ前の時刻で属性値の類似度が高いものの属性値の影響受ける
        print("一つ前の時刻で類似しているノードと同じ属性値")
        norm = attr_before.norm(dim=1)[:, None] + 1e-8
        attr_before_norm = attr_before.div(norm)
        sim = torch.matmul(attr_before_norm,torch.t(attr_before_norm))
        sim_index = torch.where(sim>0,1,0)
        sim_index = sim_index.to(torch.float32)
        sim_attr = torch.matmul(sim_index,attr_before)
        print(torch.sum((attr>0)&(sim_attr>0),dim=1))

        #全体で流行の属性値の影響を受けてる説
        #属性値を行方向に加算を行う。閾値以上のものとの一致度調べる
        #for n in range(int((torch.max(torch.sum(attr_before,dim=0)).item()))):
            
        #    print("流行の属性値 {}より大きい".format(n),torch.sum(torch.sum(attr_before,dim=0)>n))
        #    trend = (torch.sum(attr_before,dim=0)>n).repeat(32,1)
        #    trend = torch.where(trend>0,1,0)
        #    print("流行の属性値",torch.sum((attr>0)&(trend),dim=1))
           # "0より大きい(少しでも被ってる)時に日隣接の属性値と類似している.
            #さらに、この中で隣接と同じもの非隣接と同じものを探そう
            #隣接
            #print("流行と隣接の和集合",torch.sum((attr>0)&(trend)&(adj>0),dim=1))
            #非隣接
            #print("流行と非隣接の和集合",torch.sum((attr>0)&(trend)&(un_adj>0),dim=1))

        #print("流行の属性値 {}より大きい".format(0),torch.sum(torch.sum(attr_before,dim=0)>0))
        #mm = MinMaxScaler()
        ta = (torch.sum(attr_before,dim=0) - torch.min(torch.sum(attr_before,dim=0)))/(torch.max(torch.sum(attr_before,dim=0)) -torch.min(torch.sum(attr_before,dim=0)))
        print(ta)
        trend = (ta).repeat(500,1)

        #trend = torch.where(trend>0,1,0)
        ta = (attr>0)&(trend>0)
        
        print("{}流行の属性値".format(time),torch.sum(ta,dim=1))
        #trend = torch.sum(attr_before,dim=0).repeat(500,1)
        print("ysoku",torch.sum(torch.where((attr_before+adj+trend)>0.3,1,0),dim=1))
        print("総数",torch.sum(attr,dim=1))
        #for k in range(10):
           # print("{}流行の属性値".format(k),torch.sum((attr>0)&(trend>(k**2)),dim=1))
        #2hop
        #twohop_edge = torch.mm(edge_before,edge_before)
        #adj = torch.matmul(twohop_edge,attr_before)
        #print("{}2ホップ先の属性値".format(time),torch.sum((attr>0)&(adj>0),dim=1))
        




        #print(torch.sum((adj>0)&(sim_attr>0),dim=1))
        #print(torch.sum((un_adj>0)&(sim_attr>0),dim=1))

        
        #print(sim[0])#高くても0.1程度 → 閾値で分ける


        #仮説2 全体の属性値の流行の影響を受ける

