# Standard Library
import gc

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import optuna

# First Party Library
import csv
import config
from env import Env
from init_real_data import init_real_data

#another file
from memory import Memory
from actor import Actor


device = config.select_device

class PPO:
    def __init__(self, obs,agent_num,input_size, action_dim, lr, gamma,T,e,r,w,rik,temperature,story_count,data_set):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = input_size
        self.gamma = gamma
        self.persona = rik
        self.story_count = story_count 
        self.memory = Memory(agent_num, action_dim,self.story_count,data_set)
        self.obs = obs
        self.alpha = self.obs.alpha
        self.beta = self.obs.beta
        self.gamma = self.obs.beta
        self.ln = 0
        self.count = 0
        self.actor = Actor(T,e,r,w,rik,self.agent_num,temperature)
        self.new_actor = Actor(T,e,r,w,rik,self.agent_num,temperature)
        #adamにモデルを登録
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr) 
        self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr) 


    def get_actions(self, edges,feat,time):
        """ 行動確率と属性値を計算する
        Input:エッジ,属性値
        Output:
            prob エッジ方策確率
            feat 属性値
        """
        prob,feat = self.actor.predict(feat,edges,time)
        return prob,feat
    
    

    
    def train(self,gamma):
        """ 訓練用の関数
        Input:割引率
        Output:
            
        """
        
        cliprange=0.2
        storycount = self.story_count
        lnpxz = self.memory.probs.view(-1).sum()
    
        G_r = torch.empty([storycount, self.agent_num, self.agent_num])
   
        #self.memory 10x32x1
        #収益の計算
        baseline = torch.empty([storycount, self.agent_num, 1])

        for r in range(len(self.memory.reward)-1, -1, -1):

            if r == len(self.memory.reward) - 1:
                G_r[r] = self.memory.reward[r].clone()

            else:
                G_r[r] = gamma*G_r[r+1].clone() + self.memory.reward[r].clone()
       

        #baselineの作成
        for i in range(len(self.memory.reward)):
            #print( "f",self.memory.reward[i])
            
            if i == 0:
                baseline[i] = self.memory.reward[i].clone()
                #G_t[i] = self.memory.reward[i].clone()
            
            else:
                baseline[i] = (((baseline[i-1]*i)+self.memory.reward[i])/(i+1)).clone()
                #G_t[i] = gamma*G_t[i-1].clone()+self.memory.reward[i].clone()

        #print(n_v.size()) 10x1
        #print("aa",self.memory.reward.size()) 10x32x1
        #slf.,memory.reward 10x32x1
        #r1x32x1
        losses = []
        for i in range(3):
            G,loss = 0,0
            for i in range(storycount):
        
                old_policy = self.memory.probs[i]   
                new_policy,_ =  self.new_actor.forward(self.memory.features[i],self.memory.edges[i],i)
                ratio =torch.exp(torch.log(new_policy+1e-7) - torch.log(old_policy+1e-7))
                ratio_clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
                #G = G_r[i] - baseline[i]
                G = G_r[i] - baseline[i]
                reward_unclipped = ratio * G
                reward_clipped = ratio_clipped * G
                reward = torch.min(reward_unclipped, reward_clipped)
                # 最大化のために-1を掛ける
                loss = loss - torch.sum(torch.log(new_policy[i])*reward)

            self.new_actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)
                        # 勾配の計算と適用
            
            for param in self.new_actor.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data / (param.grad.data.norm() + 1e-6)

            #for name, param in self.new_actor.named_parameters():
            #    if param.grad is not None:
            #        print(f"{i}:{name} grad: {param.grad}")
            #    else:
            #        print(f"{i}:{name} grad is None")
      
            self.new_actor_optimizer.step()
            losses.append(loss)
        
        del G_r,baseline
        gc.collect()
        
        return self.new_actor.T,self.new_actor.e,self.new_actor.r,self.new_actor.W
    

def e_step(agent_num,load_data,T,e,r,w,persona,step,base_time,temperature):

    actor = Actor(T,e,r,w,persona,agent_num,temperature)
    policy_ration = torch.empty(step,len(persona[0][0]),agent_num,agent_num)
    polic_prob = actor.calc_ration(
                load_data.feature[base_time].clone(),
                load_data.adj[base_time].clone(),
                persona
                )
     
    policy_ration = polic_prob

    #時間に対して縮約 5,4,32,32 -> 5,4,32,32
    top = policy_ration

    #分子　全ての時間　 あるペルソナに注目
    rik = torch.empty(step,agent_num,len(persona[0][0]))

    #分母 top時間に縮約、bottom:ペルソナについて縮約 5,4,32,32 ->5,1,32,32
    top = torch.where(top > 0.0, top, 1e-4)
    bottom = torch.sum(top,dim=1,keepdim=True)
    top = torch.sum(top,dim=3)
    bottom= torch.sum(bottom,dim=3)
    # ration 5,4,32,32

    ration = top/(bottom)

    #ration = torch.div(top,bottom)
    # ぎょう方向に縮約
    #print("行動単位のration",ration
    #5,4,32,32 -> 5,4,32,1
    #ration = torch.mean(ration,dim=3)

    for m in range(step):
        for n in range(agent_num):
            for l in range(len(persona[0][0])):
                rik[m,n,l] = ration[m,l,n]
    
    return rik


def objective(trial):
    data_name = "NIPS"
    persona_num = 4
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
    #alpha,betaの読み込み
    if data_name == "NIPS":
        action_dim = 32
    else:
        action_dim = 500

    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    story_count = 5
    load_data = init_real_data()
    agent_num = len(load_data.adj[LEARNED_TIME])
    input_size = len(load_data.feature[LEARNED_TIME][1])

    path_n = "gamma/{}/".format(data_name)
    path = path_n+"gamma{}.npy".format(int(persona_num))
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    #5,32,4
    persona_ration = torch.from_numpy(np.tile(persona_ration,(story_count,1,1))).to(device)

    path = path_n+"means{}.npy".format(int(persona_num))
    means = np.load(path)
    means = means.astype("float32")
    means = torch.from_numpy(means).to(device)

    #重み(固定値)
    alpha = means[:,0]
    beta = means[:,1]
    gamma = means[:,2]

    print("means",means)
    print("alpha",alpha)
    print("beta",beta)
    print("gamma",gamma)

    #パラメータ
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    mu = trial.suggest_float("mu", 0.7, 0.98, log=True)
    temperature = trial.suggest_float("tem", 1e-4, 1, log=True)
    r = trial.suggest_float("r", 0.5, 1, log=True)
    w = trial.suggest_float("w", 0.5, 1, log=True)



    T = torch.tensor([1.0 for _ in range(persona_num)], dtype=torch.float32)
    e = torch.tensor([1.0 for _ in range(persona_num)], dtype=torch.float32)
    r = torch.tensor([r for _ in range(persona_num)], dtype=torch.float32)
    w = torch.tensor([w for _ in range(persona_num)], dtype=torch.float32)
  

    ln = 0
    ln_sub = 0
    sub_ln = []
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
                load_data=load_data,
                T=T,
                e=e,
                r=r,
                w=w,
                persona=persona_ration,
                step = story_count,
                base_time=LEARNED_TIME,
                temperature=temperature
                )
                      
            # スムージングファクター
            #clip_ration = 0.1
            #updated_prob_tensor = (1 - clip_ration) * mixture_ratio + clip_ration * new_mixture_ratio
            #mixture_ratio = updated_prob_tensor
            mixture_ratio = new_mixture_ratio

        #personaはじめは均等
        if episode == 0:
                    #環境の設定
            obs = Env(
                agent_num=agent_num,
                edges=load_data.adj[LEARNED_TIME].clone(),
                feature=load_data.feature[LEARNED_TIME].clone(),
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                persona=mixture_ratio
            )

        else:
            obs.reset(
                    load_data.adj[LEARNED_TIME].clone(),
                    load_data.feature[LEARNED_TIME].clone(),
                    persona=mixture_ratio
                    )
            
        
        agents = PPO(obs,agent_num, input_size, action_dim,lr, mu,T,e,r,w,mixture_ratio,temperature,story_count,data_name)
        episode_reward = 0

        for time in range(story_count):

            edge,feature = obs.state()
            edge_probs,feat_action= agents.get_actions(edge,feature,time)
            edge_action = edge_probs.bernoulli()
            #属性値を確率分布の出力と考えているので、ベルヌーイ分布で値予測
    
            reward = obs.step(feat_action,edge_action,time)
            agents.memory.probs[time]=edge_probs.clone()
            agents.memory.edges[time] = edge.clone()
            agents.memory.features[time] = feature.clone()

            agents.memory.next_edges[time]=edge_action.clone()
            agents.memory.next_features[time]=feat_action.clone()
            agents.memory.reward[time]=reward.clone()
            episode_reward = episode_reward + reward.sum()


        episodes_reward.append(episode_reward)
        print("epsiode_rewaerd",episodes_reward[-1])
        T,e,r,w= agents.train(mu)
        print("パラメータ",T,e,r,w)

    

        ln_before = ln
        ln = agents.ln
        ln_sub =abs(ln-ln_before)
        episode += 1
        sub_ln.append([ln_sub,episode_reward])
        print("ln_sub---------------------------------",ln_sub)
   
    
     
        if episode % 10 == 0:
            #print(reward)
            print(episodes_reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")

        if episode >=50:
            flag = False
 


    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    print("学習終了")
    print("パラメータ",agents.new_actor.T,agents.new_actor.e,agents.new_actor.r,agents.new_actor.W)

            
    new_mixture_ratio = e_step(
                agent_num=agent_num,
                load_data=load_data,
                T=T,
                e=e,
                r=r,
                w=w,
                persona=persona_ration,
                step = story_count,
                base_time=LEARNED_TIME,
                temperature=temperature
            )

                      

    # スムージングファクター
    #a = 0.1
    #print("nm",new_mixture_ratio)
    #updated_prob_tensor = (1 - a) * mixture_ratio + a * new_mixture_ratio
    mixture_ratio = new_mixture_ratio
    agents = PPO(obs,agent_num, input_size, action_dim,lr, gamma,T,e,r,w,mixture_ratio,temperature,story_count,data_name)

    edge_auc = 0
    attr_auc = 0
    attr_auc_log = []
    edge_auc_log = []
        
    for count in range(10):

        obs.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
            persona=mixture_ratio
        )
   
        for test_time in range(TOTAL_TIME - GENERATE_TIME):

            edges, feature = obs.state()
            edge_prob ,feat_prob ,feat_action = agents.actor.test(edges,feature,test_time)
            reward = obs.step(feat_action,edge_prob.bernoulli(),test_time)

            #属性値の評価 
            target_prob = torch.ravel(feat_prob).to("cpu")
            target_prob = target_prob.to("cpu").detach().numpy()

            detach_attr = (
                torch.ravel(load_data.feature[GENERATE_TIME + test_time])
                .detach()
                .to("cpu")
            )

            detach_attr[detach_attr > 0] = 1.0
            pos_attr = detach_attr.numpy()
            attr_test = np.concatenate([pos_attr], 0)
            
            print("pred feat sum",target_prob.sum())
            print("target feat sum",pos_attr.sum())

            attr_predict_probs = np.concatenate([target_prob], 0)

            try:
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_test),
                )
             
                auc_actv = roc_auc_score(attr_test, attr_predict_probs)
                attr_auc += auc_actv/ (TOTAL_TIME - GENERATE_TIME)
                attr_auc_log.append(auc_actv)
    
            finally:
                print("attr auc, t={}:".format(test_time), auc_actv)
                #print("attr nll, t={}:".format(t), error_attr.item())
                attr_calc_log[count][test_time] = auc_actv
                attr_calc_nll_log[count][test_time] = error_attr.item()
            del (
                target_prob,
                pos_attr,
                attr_test,
                attr_predict_probs,
                auc_actv,
            )
            gc.collect()

            #エッジの評価

            #予測データ
            edge_test= edge_prob
            #print("pi",pi_test)     
            edge_test = edge_test.view(-1)
            target_prob = edge_test.to("cpu").detach().numpy()
            edge_predict_probs = np.concatenate([target_prob], 0)

            #テストデータ
            detach_edge = (
                torch.ravel(load_data.adj[GENERATE_TIME + test_time])
                .detach()
                .to("cpu")
            )

            pos_edge = detach_edge.numpy()
            edge_test = np.concatenate([pos_edge], 0)

            criterion = nn.CrossEntropyLoss()
            error_edge = criterion(
                torch.from_numpy(edge_predict_probs),
                torch.from_numpy(edge_test),
            )
            auc_actv = roc_auc_score(edge_test, edge_predict_probs)
            edge_auc += auc_actv / (TOTAL_TIME - GENERATE_TIME)
            edge_auc_log.append(auc_actv)
  
            #print("-------")
            
            #print("edge nll, t={}:".format(t), error_edge.item())
            #print("-------")

            agents.memory.clear()
            print((edge_auc + attr_auc) / 2)

        del obs,agents,mixture_ratio

        return  (edge_auc + attr_auc) / 2


  


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # Use "minimize" if you want to minimize the objective
    study.optimize(objective, n_trials=50)
    print(study.best_params)  # Print the best hyperparameters
