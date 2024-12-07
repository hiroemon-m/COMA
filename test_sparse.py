# Standard Library
import gc

# Third Party Library

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# First Party Library
import csv
import config
from env_sparse import Env
from init_real_data import init_real_data
import time

#another file
from actor_sparse import Actor


device = config.select_device

class PPO:
    def __init__(self, obs,agent_num,input_size, action_dim, lr, gamma,T,e,r,w,rik,temperature,story_count,data_set):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = input_size
        self.gamma = gamma
        self.persona = rik
        self.story_count = story_count 
        self.obs = obs
        self.alpha = self.obs.alpha
        self.beta = self.obs.beta
        self.gamma = self.obs.gamma
        self.count = 0
        self.actor = Actor(T,e,r,w,rik,self.agent_num,temperature)
        self.new_actor = Actor(T,e,r,w,rik,self.agent_num,temperature)
        #adamにモデルを登録
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr) 
        self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr) 

    #@profile
    def get_actions(self, edges,feat,time,action_dim,feat_size):
        """ 行動確率と属性値を計算する
        Input:エッジ,属性値
        Output:
            prob エッジ方策確率
            feat 属性値
        """
        prob,edge,feat = self.actor.predict(feat,edges,time,action_dim,feat_size)
        return prob,edge,feat
    
    

    #@profile
    def train(self, gamma, action_dim, feat_size, prob_sparse_memory, edge_sparse_memory,
            feat_sparse_memory, reward_memory):
        a = time.time()
        cliprange = 0.2
        storycount = self.story_count

        with torch.no_grad():
            
            #G_r = torch.zeros_like(torch.stack(reward_memory))
            G_r = [None]*len(reward_memory)
    
            print(reward_memory[-1])
            G_r[-1] = reward_memory[-1]
    

            # 収益の計算
            for r in range(len(reward_memory) - 2, -1, -1):
                G_r[r] = gamma * G_r[r + 1] + reward_memory[r]
            
            # baselineの作成
            #baseline = torch.cumsum(reward_memory,dim=0)/torch.arange(1, len(reward_memory) + 1, 1)

        # メインループ
        for epoch in range(2):
            loss = 0
            for i in range(storycount):
                old_policy = prob_sparse_memory[i].detach().clone() 
                new_policy = self.new_actor.forward(feat_sparse_memory[i], edge_sparse_memory[i], i, action_dim, feat_size)

                # ratio計算
                #ratio = torch.exp(torch.log(new_policy.to_dense() + 1e-10) - torch.log(old_policy.to_dense() + 1e-10))
                old_log = torch.log(old_policy.values()+1e-10)
                old_ration_log = torch.sparse_coo_tensor(old_policy.indices(),old_log,old_policy.size()).coalesce()
                new_log = torch.log(new_policy.values()+1e-10)
                new_ration_log = torch.sparse_coo_tensor(new_policy.indices(),new_log,new_policy.size()).coalesce()
                ratio = torch.exp(new_ration_log.values() - old_ration_log.values())
                print(ratio.sum())

                new_ration_log_indices = new_ration_log.indices()
                old_ration_log_indices = old_ration_log.indices()
                new_indices = torch.cat([new_ration_log_indices, old_ration_log_indices], dim=1)
                
                non_zero_indices = ratio.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
                filtered_indices = new_indices[:, non_zero_indices]  # 0でないインデックスのみ
                filtered_values = ratio[non_zero_indices]  # 0でない値のみ 
                ratio = torch.sparse_coo_tensor(filtered_indices,filtered_values,new_ration_log.size()).coalesce()


                ratio_clipped = torch.clamp(ratio.to_dense(), 1 - cliprange, 1 + cliprange)

                #G = G_r[i] - baseline[i]
                G = G_r[i] 

                # 報酬計算
                print(G.size())
                print(ratio.to_sparse().size())
                reward_unclipped = ratio.to_sparse().multiply(G)
                #reward_clipped = ratio_clipped.to_sparse() * G
                #reward = torch.min(reward_clipped.to_dense(),reward_unclipped.to_dense())
        
                # 新しいポリシーの損失計算
                reward = reward_unclipped
                print(torch.sum(new_policy))
                non_zero_indices = new_policy.values().nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
                filtered_indices = new_policy.indices()[:, non_zero_indices]  # 0でないインデックスのみ
                filtered_values = new_policy.values()[non_zero_indices]  # 0でない値のみ 
                new_policy = torch.sparse_coo_tensor(filtered_indices,filtered_values,new_policy.size()).coalesce()

                policy_values = torch.log(new_policy.values())
                print(torch.sum(policy_values))
                new_policy_coo = torch.sparse_coo_tensor(new_policy.indices(), policy_values, new_policy.size())
                print(torch.sparse.sum(new_policy_coo))
                loss -= torch.sparse.sum(new_policy_coo * reward)

                # メモリ解放
                del old_policy, new_policy, ratio, ratio_clipped, reward_unclipped,  reward, G
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

            # 勾配の適用
            self.new_actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)  # retain_graph=False でグラフを保持しない
            torch.nn.utils.clip_grad_norm_(self.new_actor.parameters(), 5)
            self.new_actor_optimizer.step()

            # 勾配情報のログ出力
            for name, param in self.new_actor.named_parameters():
                if param.grad is not None:
                    print(f"{epoch}:{name} grad: {param.grad}")
                else:
                    print(f"{epoch}:{name} grad is None")

            print("update", self.new_actor.T, self.new_actor.e, self.new_actor.r, self.new_actor.W)

        # パラメータのコピーとメモリ解放
        T, e, r, w = self.new_actor.T.clone().detach(), self.new_actor.e.clone().detach(), self.new_actor.r.clone().detach(), self.new_actor.W.clone().detach()

        # 不要な変数の削除
        del G_r, loss, self.new_actor, self.new_actor_optimizer, self.actor
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

        b = time.time()
        print("train",b-a)

        return T, e, r, w

    
#@profile
def e_step(agent_num,edge,feat,T,e,r,w,persona,step,temperature,sparse_size):
    a = time.time()
    #csr
    actor = Actor(T,e,r,w,persona,agent_num,temperature)
    #5,5 sparse tensorが入っている

    policy_prob = actor.calc_ration(
                feat,
                edge,
                persona,
                sparse_size
                )
    



    #分子　全ての時間　 あるペルソナに注目
    rik = torch.empty(step,agent_num,len(persona[0][0]))
    ration = torch.empty(step,len(persona[0][0]),agent_num)

    #分母 top時間に縮約、bottom:ペルソナについて縮約 5,4,32,32 ->5,1,32,32
    top = policy_prob
    bottom_sum = []
    for row in policy_prob:
        bottom_tensor = row[0]
        for tensor in row[1:]:
            bottom_tensor = bottom_tensor + tensor
        bottom_sum.append(bottom_tensor)

    
    # ration 5,4,32,32
    for j in range(5):
        for k in range(len(persona[0][0])):
            ration[j][k] = (torch.sum(top[j][k].to_dense(),dim=-1)/torch.sum((bottom_sum[k].to_dense()),dim=-1))

    #ration = top/(bottom_sum)


    #ration = torch.div(top,bottom)
    # ぎょう方向に縮約
    #print("行動単位のration",ration
    #5,4,32,32 -> 5,4,32,1
    #ration = torch.mean(ration,dim=3)


    for m in range(step):
        for n in range(agent_num):
            for l in range(len(persona[0][0])):
                original = ration[m,l,n]
                rik[m,n,l] = original.detach()
    

    del actor,policy_prob,top,ration,original
    gc.collect()
    b = time.time()
    print("e-step",b-a)
    return rik
    

#@profile
def show():
    print("dell")

#@profile
def execute_data(persona_num,data_name,data_type):
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
    #alpha,betaの読み込み
    if data_name == "NIPS":
        action_dim = 32

    if data_name == "DBLP":
        action_dim = 500

    if data_name == "Twitter":
        action_dim = 112044

    if data_name == "Reddit":
        action_dim = 8077
    

    if data_name == "NIPS":
        feat_size = 2411
    if data_name == "DBLP":
        feat_size = 3854
    if data_name == "Twitter":
        feat_size = 5372
    if data_name == "Reddit":
        feat_size = 300

    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    story_count = 5
    load_data = init_real_data(data_name)

    edge_sparse = []
    feat_sparse = []
    for i in range(LEARNED_TIME+1):
        edge_sparse.append(load_data.adj[i])
        feat_sparse.append(load_data.feature[i])
    

    agent_num = len(load_data.adj[LEARNED_TIME])
    input_size = len(load_data.feature[LEARNED_TIME][1])

    path_n = "optimize/{}/{}/".format(data_type,data_name)
    path = path_n+"persona={}/gamma.npy".format(int(persona_num))
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    #time,node,persona
    persona_ration = torch.from_numpy(np.tile(persona_ration,(story_count,1,1))).to(device)

    path = path_n+"persona={}/means.npy".format(int(persona_num))
    means = np.load(path)
    means = means.astype("float32")
    means = torch.from_numpy(means).to(device)

    #重み(固定値)
    alpha = means[:,0]
    beta = means[:,1]
    gamma = means[:,2]
    
    print("load",load_data.adj)
    print("load",load_data.feature)

    print("means",means)
    print("alpha",alpha)
    print("beta",beta)
    print("gamma",gamma)

    #パラメータ
    if data_name == "NIPS":
        mu = 0.194
        lr = 1.563e-06
        lr = 0.001
        temperature = 0.01
        T = torch.tensor([1.055 for _ in range(persona_num)], dtype=torch.float32)
        e = torch.tensor([1.347 for _ in range(persona_num)], dtype=torch.float32)
        r = torch.tensor([0.697 for _ in range(persona_num)], dtype=torch.float32)
        w = torch.tensor([0.026 for _ in range(persona_num)], dtype=torch.float32)
    
    elif data_name == "DBLP":
        mu = 0.0229
        lr = 0.000952
        temperature = 0.01
        T = torch.tensor([1.481 for _ in range(persona_num)], dtype=torch.float32)
        e = torch.tensor([0.759 for _ in range(persona_num)], dtype=torch.float32)
        r = torch.tensor([0.868 for _ in range(persona_num)], dtype=torch.float32)
        w = torch.tensor([0.846 for _ in range(persona_num)], dtype=torch.float32)
  
    else:
        mu = 0.01
        lr = 0.00001
        temperature = 0.01
        T = torch.tensor([0.70 for _ in range(persona_num)], dtype=torch.float32)
        e = torch.tensor([0.1 for _ in range(persona_num)], dtype=torch.float32)
        r = torch.tensor([0.80 for _ in range(persona_num)], dtype=torch.float32)
        w = torch.tensor([0.80 for _ in range(persona_num)], dtype=torch.float32)

    ln = 0
    ln_sub = 0

    flag = True
    episode = 0
    episodes_reward = []

    while flag and ln_sub <= 1:
        
        print("----------episode:{}----------".format(episode))

        # E-step
        #mixture_ratio:混合比率

        if episode == 0:
            mixture_ratio = persona_ration
       
        else:
            new_mixture_ratio = e_step(
                agent_num=agent_num,
                edge = edge_sparse[LEARNED_TIME],
                feat = feat_sparse[LEARNED_TIME],
                T=T,
                e=e,
                r=r,
                w=w,
                persona=persona_ration,
                step = story_count,
                temperature=temperature,
                sparse_size = feat_size
                )
     
                      
            # スムージングファクター
            if episode <= 3:
                clip_ration = 0.2
                updated_prob_tensor = (1 - clip_ration) * mixture_ratio + clip_ration * new_mixture_ratio
                mixture_ratio = updated_prob_tensor
                del updated_prob_tensor
                gc.collect()
            else:
                mixture_ratio = new_mixture_ratio


        #personaはじめは均等
        
                    #環境の設定
        obs = Env(
            agent_num=agent_num,
            edge = edge_sparse[LEARNED_TIME].clone(),
            feature = feat_sparse[LEARNED_TIME].clone(),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            persona=mixture_ratio
        )
        print("ちん",edge_sparse[LEARNED_TIME].size())
        print(feat_sparse[LEARNED_TIME].size())
        print(mixture_ratio.size())

        #else:
        #    obs.reset(
        #            load_data.adj[LEARNED_TIME].clone(),
        #            load_data.feature[LEARNED_TIME].clone(),
        #            persona=mixture_ratio
        #            )
            
        
        agents = PPO(obs,agent_num, input_size, action_dim,lr, mu,T,e,r,w,mixture_ratio,temperature,story_count,data_name)
        episode_reward = 0
        print("persona_ration",mixture_ratio)
        prob_sparse_memory = []
        #next_edge_sparse_memory = []
        #next_feat_sparse_memory = []
        edge_sparse_memory = []
        feat_sparse_memory = []
        reward_memory = []
        edge_sparse_memory.append(edge_sparse[LEARNED_TIME].clone())
        feat_sparse_memory.append(feat_sparse[LEARNED_TIME].clone())

        for time in range(story_count):

            edge,feature = obs.state()
    
            #edge_probs COO
            print("before get actions")
            edge_probs,edge_action,feat_action= agents.get_actions(edge,feature,time,action_dim,feat_size)
            print("done get actions")

            #属性値を確率分布の出力と考えているので、ベルヌーイ分布で値予測
    
            reward = obs.step(feat_action,edge_action,time)
            print("done reward")
            print(reward.size())

            #agents.memory.probs[time]=edge_probs.clone()
            prob_sparse_memory.append(edge_probs.clone())

            #agents.memory.edges[time] = edge.clone()
            edge_sparse_memory.append(edge_action.clone())
            #agents.memory.features[time] = feature.clone()
            feat_sparse_memory.append(feat_action.clone())

            #print("pred_feat",torch.sum(feat_action))
            #print("pred_edge",torch.sum(edge_action))

            #agents.memory.next_edges[time]=edge_action.clone()
            #next_edge_sparse_memory.apeend(edge_action.clone())
            #agents.memory.next_features[time]=feat_action.clone()
            #next_feat_sparse_memory.apeend(feat_action.clone())

            #agents.memory.reward[time]=reward.clone()
            reward_memory.append(reward.clone())
            episode_reward = episode_reward + torch.sparse.sum(reward)
            del edge,feature,edge_probs,edge_action,feat_action,reward
            gc.collect()
      

        episodes_reward.append(episode_reward)
        print(reward_memory)
        #print("epsiode_rewaerd",episodes_reward[-1])
        print("go train")
        T,e,r,w= agents.train(mu,action_dim,feat_size,prob_sparse_memory,edge_sparse_memory,
                              feat_sparse_memory,reward_memory)
        print("done train")
        


        #print("persona_ration",mixture_ratio)


        #ln_before = ln
        #ln = agents.ln
        #ln_sub =abs(ln-ln_before)
        episode += 1
        #sub_ln.append([ln_sub,episode_reward])
        #print("ln_sub---------------------------------",ln_sub)
   
    
     
        if episode % 10 == 0:
            #print(reward)
            print(episodes_reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")

        if episode >=10:
            flag = False

        else:
            print("delete")
            show()
            del episode_reward
            del agents,obs
            gc.collect()
            show()
 


    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    print("学習終了")
    print("パラメータ",T,e,r,w)

            
    new_mixture_ratio = e_step(
                agent_num=agent_num,
                edge = edge_sparse[LEARNED_TIME],
                feat = feat_sparse[LEARNED_TIME],
                T=T,
                e=e,
                r=r,
                w=w,
                persona=persona_ration,
                step = story_count,
                temperature=temperature,
                sparse_size = feat_size
            )

                      

    # スムージングファクター
    #
    # a = 0.1
    #print("nm",new_mixture_ratio)
    #updated_prob_tensor = (1 - a) * mixture_ratio + a * new_mixture_ratio
    mixture_ratio = new_mixture_ratio
    agents = PPO(obs,agent_num, input_size, action_dim,lr, gamma,T,e,r,w,mixture_ratio,temperature,story_count,data_name)


    for count in range(10):

        obs.reset(
            edge_sparse[LEARNED_TIME].clone(),
            feat_sparse[LEARNED_TIME].clone(),
            persona=mixture_ratio
        )
   
        for test_time in range(TOTAL_TIME - GENERATE_TIME):

            edges, feature = obs.state()
            edge_action,edge_prob ,feat_prob ,feat_action = agents.actor.test(edges,feature,test_time,action_dim,feat_size)
            reward = obs.step(feat_action,edge_action,test_time)
            

            #属性値の評価 
            pred_prob = torch.ravel(feat_prob.to_dense()).to("cpu")
            pred_prob = pred_prob.to("cpu").detach().numpy()
            dense_array = np.array(load_data.feature[GENERATE_TIME + test_time].todense())
            attr_tensor = torch.from_numpy(dense_array)

            detach_attr = (
                torch.ravel(attr_tensor)
                .detach()
                .to("cpu")
            )
            
            gc.collect()

            detach_attr[detach_attr > 0] = 1.0
            pos_attr = detach_attr.numpy()
            attr_test = np.concatenate([pos_attr], 0)
            attr_predict_probs = np.concatenate([pred_prob], 0)
            print("属性値の総数")
        
            #print("pred feat sum",attr_predict_probs.reshape((500, -1)).sum(axis=1))
            #print("target edge sum",attr_test.reshape((500, -1)).sum(axis=1))
            print("pred feat sum",attr_predict_probs.sum())
            print("feat action sum",feat_action.sum())
            print("target edge sum",attr_test.sum())
            


            try:
                
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_test),
                )
               
               
              
                auc_actv = roc_auc_score(attr_test, attr_predict_probs)
 
            
            finally:
                print("attr auc, t={}:".format(test_time), auc_actv)
                #print("attr auc, t={}:".format(test_time), pr_auc)
                #print("attr nll, t={}:".format(t), error_attr.item())
                #attr_calc_log[count][test_time] = auc_actv
                attr_calc_log[count][test_time] = auc_actv
                attr_calc_nll_log[count][test_time] = error_attr.item()
            
      
            del pred_prob,detach_attr
            #エッジの評価

            #予測データ
            target_prob= edge_prob.to_dense()
            #print("pi",pi_test)     
            target_prob = target_prob.view(-1)
            target_prob = target_prob.to("cpu").detach().numpy()
            edge_predict_probs = np.concatenate([target_prob], 0)
            
            #テストデータ
            dense_array = np.array(load_data.adj[GENERATE_TIME + test_time].to_dense())
            edge_tensor = torch.from_numpy(dense_array)
            detach_edge = (
                torch.ravel(edge_tensor)
                .detach()
                .to("cpu")
            )

            pos_edge = detach_edge.numpy()
            edge_test = np.concatenate([pos_edge], 0)

            print("エッジの総数")
            #print("pred feat sum",edge_predict_probs.reshape((500, -1)).sum(axis=1))
            #print("target edge sum",edge_test.reshape((500, -1)).sum(axis=1))
            print("pred edge sum",edge_predict_probs.sum())
            print("acction edge sum",edge_action.sum())
            print("target edge sum",edge_test.sum())

            criterion = nn.CrossEntropyLoss()
            error_edge = criterion(
                torch.from_numpy(edge_predict_probs),
                torch.from_numpy(edge_test),
            )
            del target_prob,detach_edge
            gc.collect()

            auc_calc = roc_auc_score(edge_test, edge_predict_probs)  
            print("edge auc, t={}:".format(test_time), auc_calc)

            print(T,e,r,w)
            calc_log[count][test_time] = auc_calc
            calc_nll_log[count][test_time] = error_edge.item()
           

            path_save = "experiment_data"

            np.save(path_save+"/proposed_edge_auc", calc_log)
            np.save(path_save+"/proposed_edge_nll", calc_nll_log)
            np.save(path_save+"/proposed_attr_auc", attr_calc_log)
            np.save(path_save+"/proposed_attr_nll", attr_calc_nll_log)
            #print("t",T,"e",e,"r",r,"w",w)
            #print(mixture_ratio)
            np.save(path_save+"/persona_ration",np.concatenate([mixture_ratio.detach().numpy()],axis=0))
            np.save(path_save+"/paramerter",np.concatenate([T.detach().numpy(),e.detach().numpy(),r.detach().numpy(),w.detach().numpy()],axis=0))

        




if __name__ == "__main__":
    #[5,8,12,16,24,32,64,128]
    #[4,8,12,16]
    s = time.time()
    for i in [5]:
        execute_data(i,"Twitter","complete")
    e = time.time()
    print("time:",s-e)