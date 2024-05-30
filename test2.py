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
    def __init__(self, obs,agent_num,input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,rik,story_count):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = input_size
        self.gamma = gamma
        self.persona = rik
        self.target_update_steps = target_update_steps
        self.story_count = story_count 
        self.memory = Memory(agent_num, action_dim,self.story_count)
        self.obs = obs
        self.alpha = self.obs.alpha
        self.beta = self.obs.beta
        self.ln = 0
        self.count = 0
        self.actor = Actor(T,e,r,w,rik)
        self.new_actor = Actor(T,e,r,w,rik)


        #adamにモデルを登録
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a) 
        self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr_a) 
        




    def get_actions(self, edges,feat,time):
        
        prob,feat = self.actor.predict(feat,edges,time)
    
        return feat,prob
    
    

    
    def train(self,gamma):

        #----G(t)-b(s)----
        G,loss = 0,0
        cliprange=0.2
        storycount = self.story_count

        lnpxz = self.memory.probs.view(-1).sum()
        G_r = torch.empty([storycount,32,32])

        #baselineの作成
        #self.memory 10x32x1

        baseline = torch.empty([storycount,32,1])

        for r in range(len(self.memory.reward) - 1,-1,-1):
            if r == len(self.memory.reward) - 1:
                G_r[r] = self.memory.reward[r].clone()

            else:
                G_r[r] = gamma*G_r[r+1].clone() + self.memory.reward[r].clone()
                #print("G",r,G[r])
                #print("after",r,self.memory.reward[r])

   


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
        for i in range(storycount):
     
            old_policy = self.memory.probs[i]

            new_policy,_ =  self.new_actor.forward(self.memory.features[i],self.memory.edges[i],i)
            ratio =torch.exp(torch.log(new_policy+1e-7) - torch.log(old_policy+1e-7))
            ratio_clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
            G = G_r[i] - baseline[i]
        
            loss_unclipped = ratio * G
            loss_clipped = ratio_clipped * G
            loss = torch.min(loss_unclipped, loss_clipped)
            # 最大化のために-1を掛ける
            loss = -loss.mean()
            #print("loss",loss)
  


            self.new_actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)
                        # 勾配の計算と適用
            

            self.new_actor_optimizer.step()
            losses.append(loss)

            #print("更新後",self.new_actor.T,self.new_actor.e,self.new_actor.r,self.new_actor.W)

            #print("loss",loss.grad)
            #print("t",self.new_actor.T.grad)
            #print("e",self.new_actor.e.grad)
            #print("r",self.new_actor.r.grad)
            #print("w",self.new_actor.W.grad)

        
        return self.new_actor.T,self.new_actor.e,self.new_actor.r,self.new_actor.W,losses
    


def e_step(agent_num,load_data,T,e,r,w,persona,step,base_time):

    actor = actor = Actor(T,e,r,w,persona)
  
    #personaはじめは均等
    policy_ration = torch.empty(step,len(persona[0][0]),agent_num,agent_num)
   

    for time in range(step):
        polic_prob = actor.calc_ration(
                    load_data.feature[base_time+time].clone(),
                    load_data.adj[base_time+time].clone(),
                    persona,
                    time
                    )
     
        policy_ration[time] = polic_prob


    #時間に対して縮約 5,4,32,32 -> 4,32,32
    top = policy_ration
    #print("top",top)

    #分子　全ての時間　 あるペルソナに注目
    rik = torch.empty(step,agent_num,len(persona[0][0]))
    #分母 top時間に縮約、bottom:ペルソナについて縮約 4,32,32 -> 32,32
    bottom = torch.sum(top,dim=1,keepdim=True)
     
    top = torch.mean(top,dim=3)
    bottom= torch.mean(bottom,dim=3)

    #print("行動単位のration",ration)
    ration = top/(bottom+1e-7)
 

    for m in range(step):
        for n in range(agent_num):
            for l in range(len(persona[0][0])):
                rik[m,n,l] = ration[m,l,n]

    return rik

import optuna

# Define the objective function for Optuna
def objective(trial):
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
    
    # Define the hyperparameters to be optimized
    lr_a = trial.suggest_float('lr_a', 1e-5, 1e-1, log=True)
    lr_c = trial.suggest_float('lr_c', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.99)

    
    #alpha,betaの読み込み
    path_n = "gamma/complete/"
    persona_num = 4
    path = path_n+"gamma{}.npy".format(int(persona_num))
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    persona_ration = torch.from_numpy(persona_ration).to(device)
    
    path = path_n+"means{}.npy".format(int(persona_num))
    means = np.load(path)
    means = means.astype("float32")
    means = torch.from_numpy(means).to(device)
    alpha = means[:,0]
    beta = means[:,1]
    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    load_data = init_real_data()
    agent_num = len(load_data.adj[LEARNED_TIME])
    input_size = len(load_data.feature[LEARNED_TIME][1])
    action_dim = 32
    N = 32
    
    target_update_steps = 8
    story_count = 5

    T = torch.tensor([[trial.suggest_float(f'T_{i}', 0.7, 1.2) for i in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    e = torch.tensor([[trial.suggest_float(f'e_{i}', 0.7, 1.2) for i in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    r = torch.tensor([[trial.suggest_float(f'r_{i}', 0.7, 1.2) for i in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    w = torch.tensor([[trial.suggest_float(f'w_{i}', 0.7, 1.2) for i in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    

  



    ln = 0
    ln_sub = 0
    sub_ln = []
    s_loss = []
    flag = True
    episode = 0
    count = 0
    episodes_reward = []
    persona_ration = torch.from_numpy(np.tile(persona_ration,(story_count,1,1))).to(device)
    

    while flag and ln_sub <= 1:
        # E-step
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
                step = GENERATE_TIME,
                base_time=LEARNED_TIME
            )
            alpha = 0.1
            updated_prob_tensor = (1 - alpha) * mixture_ratio + alpha * new_mixture_ratio
            mixture_ratio = updated_prob_tensor

        if episode == 0:
            obs = Env(
                agent_num=agent_num,
                edges=load_data.adj[LEARNED_TIME].clone(),
                feature=load_data.feature[LEARNED_TIME].clone(),

                alpha=alpha,
                beta=beta,
                persona=mixture_ratio
            )
        else:
            obs.reset(
                load_data.adj[LEARNED_TIME].clone(),
                load_data.feature[LEARNED_TIME].clone(),
                persona=mixture_ratio
            )
            
        episode_reward = 0
       
        agents = PPO(obs,agent_num, input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,mixture_ratio,story_count)
        for i in range(story_count):
            edges,feature = obs.state()
            feat,action_probs = agents.get_actions(edges,feature,i)
            action = action_probs.bernoulli()
            reward = obs.step(feat,action,i)

            agents.memory.probs[i]=action_probs.clone()
            agents.memory.edges[i] = edges.clone()
            agents.memory.features[i] = feature.clone()
            agents.memory.next_edges[i]=action.clone()
            agents.memory.next_features[i]=feat.clone()
            agents.memory.reward[i]=reward.clone()
            episode_reward = episode_reward + reward.sum()

        episodes_reward.append(episode_reward)
        new_T,new_e,new_r,new_w,loss = agents.train(gamma)
        count +=1
        s_loss.append(sum(loss))
        print("loss",s_loss)
        T = new_T
        e = new_e
        r = new_r
        w = new_w
  
        
        ln_before = ln
        ln = agents.ln
        ln_sub =abs(ln-ln_before)
        episode += 1
        sub_ln.append([ln_sub,episode_reward])
        
        if episode % 10 == 0:
            print(episodes_reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")
        if episode >=50:
            flag = False
    
    # The final return value for the objective function should be the value to be maximized or minimized
    print("sum",s_loss)
    return sum(s_loss)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # Use "minimize" if you want to minimize the objective
    study.optimize(objective, n_trials=25)

    print(study.best_params)  # Print the best hyperparameters
