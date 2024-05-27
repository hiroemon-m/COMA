# Standard Library
import gc

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import tensorflow as tf




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
    def __init__(self, obs,agent_num,input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,q,w,rik,story_count):
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
        self.actor = Actor(T,e,r,q,w,rik)
        self.new_actor = Actor(T,e,r,q,w,rik)
        #adamにモデルを登録
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a) 
        self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr_a) 


    def get_actions(self, edges,feat,time):
        
        prob,feat = self.actor.predict(feat,edges,time)
                # 損失を計算する（ここでは単純にsumを取る）

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
    
            reward_unclipped = ratio * G
            reward_clipped = ratio_clipped * G
            reward = torch.min(reward_unclipped, reward_clipped)
            # 最大化のために-1を掛ける
            loss = -reward.mean()

  


            self.new_actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)
                        # 勾配の計算と適用
            

            for param in self.new_actor.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data / (param.grad.data.norm() + 1e-6)

            for name, param in self.new_actor.named_parameters():
                if param.grad is not None:
                    print(f"{i}:{name} grad: {param.grad}")
                else:
                    print(f"{i}:{name} grad is None")

      

            self.new_actor_optimizer.step()
            losses.append(loss)

            #print("更新後",self.new_actor.T,self.new_actor.e,self.new_actor.r,self.new_actor.W)

            #print("loss",loss.grad)
            #print("t",self.new_actor.T.grad)
            #print("e",self.new_actor.e.grad)
            #print("r",self.new_actor.r.grad)
            #print("w",self.new_actor.W.grad)

        
        return self.new_actor.T,self.new_actor.e,self.new_actor.r,self.new_actor.q,self.new_actor.W,
    


def e_step(agent_num,load_data,T,e,r,q,w,persona,step,base_time):

    actor = actor = Actor(T,e,r,q,w,persona)
  
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
   


    #時間に対して縮約 5,4,32,32 -> 5,4,32,32
    top = policy_ration
    #print("top",top)


    #分子　全ての時間　 あるペルソナに注目
    rik = torch.empty(step,agent_num,len(persona[0][0]))
    #分母 top時間に縮約、bottom:ペルソナについて縮約 5,4,32,32 ->5,1,32,32
    bottom = torch.sum(top,dim=1,keepdim=True)

    print(top[0,:,0,0])

    print(bottom[0,0,0,0])
    top = torch.mean(top,dim=3)
    bottom= torch.mean(bottom,dim=3)


    # ration 5,4,32,32
    ration = top/(bottom+1e-7)

    
    #ration = torch.div(top,bottom)
    # ぎょう方向に縮約
    #print("行動単位のration",ration
    #5,4,32,32 -> 5,4,32,1
    #ration = torch.mean(ration,dim=3)

    #print("ration",torch.sum(ration,dim=0))
    for m in range(step):
        for n in range(agent_num):
            for l in range(len(persona[0][0])):
                rik[m,n,l] = ration[m,l,n]

    return rik


def execute_data():
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
    #alpha,betaの読み込み
       #ペルソナの取り出し
    #ペルソナの数[3,4,5,6,8,12]
    path_n = "gamma/complete/"
    persona_num = 4
    path = path_n+"gamma{}.npy".format(int(persona_num))
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")

    


    #print(persona_ration)
    path = path_n+"means{}.npy".format(int(persona_num))
    means = np.load(path)
    means = means.astype("float32")
    means = torch.from_numpy(means).to(device)
    alpha = means[:,0]
    beta = means[:,1]
    print("means",means)
    print("alpha",alpha)
    print("beta",beta)
    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    load_data = init_real_data()
    agent_num = len(load_data.adj[LEARNED_TIME])
    input_size = len(load_data.feature[LEARNED_TIME][1])

    action_dim = 32
    N = 32
    #パラメータ
    gamma = 0.90
    lamda = 0.95
    lr_a = 0.01
    lr_c = 0.01
    target_update_steps = 8
    alpha = alpha
    beta = beta
    story_count = 5

    #5,32,4
    persona_ration = torch.from_numpy(np.tile(persona_ration,(story_count,1,1))).to(device)

    #5,4
    T = torch.tensor([[1.0 for _ in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    e = torch.tensor([[1.0 for _ in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    r = torch.tensor([[1.0 for _ in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    q = torch.tensor([[1.0 for _ in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    w = torch.tensor([[1.0 for _ in range(persona_num)] for s in range(story_count)], dtype=torch.float32)
    

    ln = 0
    ln_sub = 0
    sub_ln = []
    flag = True
    episode = 0
    count = 0
    episodes_reward = []


    while flag and ln_sub <= 1:


        # E-step
        if episode == 0:

            mixture_ratio = persona_ration
       
        else:
            print(episode)
            print("-------------------------")
            #mixture_ratio = persona_ration
            #softの時は外す
            #mixture_ratio:混合比率
            new_mixture_ratio = e_step(
                agent_num=agent_num,
                load_data=load_data,
                T=T,
                e=e,
                r=r,
                q=q,
                w=w,
                persona=persona_ration,
                step = story_count,
                base_time=LEARNED_TIME
                )
                      

            # スムージングファクター
            alpha = 0.1
            #print("nm",new_mixture_ratio)
            updated_prob_tensor = (1 - alpha) * mixture_ratio + alpha * new_mixture_ratio
            mixture_ratio = updated_prob_tensor
      

        #personaはじめは均等
        if episode == 0:
                    #環境の設定
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
        agents = PPO(obs,agent_num, input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,q,w,mixture_ratio,story_count)
        for i in range(story_count):

            edges,feature = obs.state()
            feat,action_probs = agents.get_actions(edges,feature,i)
           
            #print("0",action_probs[action_probs<0].sum())
            #print("1",action_probs[action_probs>1].sum())
            #print("nan",action_probs[action_probs=="Nan"].sum())
            #->nanいる
   
            action = action_probs.bernoulli()
            #属性値を確率分布の出力と考えているので、ベルヌーイ分布で値予測
            #feat = torch.clamp(feat,min=0)

        
            reward = obs.step(feat,action,i)
   
            # 勾配を計算する
         
            #sotry_count,agentnum,1→各行動の報酬のtonsorを保持した方が勾配計算うまくいくかも

            agents.memory.probs[i]=action_probs.clone()
            agents.memory.edges[i] = edges.clone()
            agents.memory.features[i] = feature.clone()
            agents.memory.next_edges[i]=action.clone()
            agents.memory.next_features[i]=feat.clone()
            agents.memory.reward[i]=reward.clone()
            episode_reward = episode_reward + reward.sum()
            #memory_li.append((reward,action_probs))

        episodes_reward.append(episode_reward)
        print("epsiode_rewaerd",episodes_reward[-1])
        T,e,r,q,w= agents.train(gamma)
        #new_T,new_e,new_r,new_w = agents.train_a(memory,gamma)
        print("r",r,"q",q)
        count +=1


  
     

        ln_before = ln
        ln = agents.ln
        ln_sub =abs(ln-ln_before)
        episode += 1
        sub_ln.append([ln_sub,episode_reward])
        print("ln_sub---------------------------------",ln_sub)
        episode += 1
    

   
        if episode % 10 == 0:
            #print(reward)
            print(episodes_reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")
        if episode >=100:
            flag = False
        #print("T",T,"e",e,"r",r,"w",w,"alpha",alpha,"beta",beta)
    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))
    print(sub_ln)
    print("学習後",agents.new_actor.T,agents.new_actor.e,agents.new_actor.r,agents.new_actor.q,agents.new_actor.W)

    
        
    for count in range(10):
        obs.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
            persona=persona_ration
        )
   
        for t in range(TOTAL_TIME - GENERATE_TIME):


            gc.collect()
            #field.state()隣接行列、属性値を返す
            #neighbor_state, feat = field.state()
            #->部分観測(自分のエッジの接続、属性値の情報)にする
            
            edges, feature = obs.state()
            #print("stae",neighbor_state)
            #print("feat",feat)
            #featもprobも確率
            prob ,feat ,feat_bernoulli = agents.actor.test(
              edges,feature,t
            )
            del edges, feature
           

            reward = obs.step(feat_bernoulli,prob.bernoulli(),t)

            #属性値の評価 
            target_prob = torch.ravel(feat).to("cpu")
            detach_attr = (
                torch.ravel(load_data.feature[GENERATE_TIME + t])
                .detach()
                .to("cpu")
            )
            detach_attr[detach_attr > 0] = 1.0
            pos_attr = detach_attr.numpy()
            attr_numpy = np.concatenate([pos_attr], 0)
            target_prob = target_prob.to("cpu").detach().numpy()
            print("pre",target_prob.sum())
            print("tar",pos_attr.sum())
            attr_predict_probs = np.concatenate([target_prob], 0)
            try:
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_numpy),
                )
                auc_actv = roc_auc_score(attr_numpy, attr_predict_probs)
    
            finally:
                print("attr auc, t={}:".format(t), auc_actv)
                #print("attr nll, t={}:".format(t), error_attr.item())
                attr_calc_log[count][t] = auc_actv
                attr_calc_nll_log[count][t] = error_attr.item()
            del (
                target_prob,
                pos_attr,
                attr_numpy,
                attr_predict_probs,
                auc_actv,
            )
            gc.collect()

            #エッジの評価

            #予測データ
            pi_test= prob
            #print("pi",pi_test)     
            pi_test = pi_test.view(-1)
            target_prob = pi_test.to("cpu").detach().numpy()
            edge_predict_probs = np.concatenate([target_prob], 0)

            #テストデータ
            detach_edge = (
                torch.ravel(load_data.adj[GENERATE_TIME + t])
                .detach()
                .to("cpu")
            )
            pos_edge = detach_edge.numpy()
            edge_numpy = np.concatenate([pos_edge], 0)


            criterion = nn.CrossEntropyLoss()
            error_edge = criterion(
                torch.from_numpy(edge_predict_probs),
                torch.from_numpy(edge_numpy),
            )
            auc_calc = roc_auc_score(edge_numpy, edge_predict_probs)
  
       
            #print("-------")
            print("edge auc, t={}:".format(t), auc_calc)
            #print("edge nll, t={}:".format(t), error_edge.item())
            #print("-------")


            calc_log[count][t] = auc_calc
            calc_nll_log[count][t] = error_edge.item()

            agents.memory.clear()
            
        #print("---")


    np.save("experiment_data/DBLP/abligation/persona={}/proposed_edge_auc".format(persona_num), calc_log)
    np.save("experiment_data/DBLP/abligation/persona={}/proposed_edge_nll".format(persona_num), calc_nll_log)
    np.save("experiment_data/DBLP/abligation/persona={}/proposed_attr_auc".format(persona_num), attr_calc_log)
    np.save("experiment_data/DBLP/abligation/persona={}/proposed_attr_nll".format(persona_num), attr_calc_nll_log)
    print("t",T,"e",e,"r",r,"w",w)
    np.save("experiment_data/DBLP/abligation/persona={}/parameter".format(persona_num),np.concatenate([alpha.detach(),beta.detach().numpy(),T.detach().numpy(),e.detach().numpy()],axis=0))
    np.save("experiment_data/DBLP/abligation/persona={}/rw_paramerter".format(persona_num),np.concatenate([r.detach().numpy().reshape(1,-1),w.detach().numpy().reshape(1,-1)],axis=0))


  




if __name__ == "__main__":
    execute_data()