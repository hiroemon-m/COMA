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
    def __init__(self, obs,agent_num,input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,rik,story_count):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = 2411
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
        #self.new_actor = Actor(T,e,r,w,rik)


        #adamにモデルを登録
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a) 
        #self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr_a) 
        




    def get_actions(self,env, edges,feat):
        
        #with torch.no_grad():
        
        prob,feat = self.actor.predict(feat,edges)
    
        return feat,prob
    
    

    
    def train(self,gamma,lamda,param,memory):

        #----G(t)-b(s)----
        G,loss = 0,0
    
        storycount = self.story_count

        losses = []
        G,loss = 0,0
        
        for reward,prob in reversed(memory):
            
            G = reward + lamda*G
            #print("prob",prob)
            #print("prob",G)
            loss = loss - torch.sum(torch.log(prob)*G)
            # 勾配の計算と適用
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        losses.append(loss)
        for name, param in self.new_actor.named_parameters():
            if param.grad is not None:
                print(f"{name} grad: {param.grad}")
            else:
                print(f"{name} grad is None")

        #print("loss",loss.grad)
        #print("t",self.new_actor.T.grad)
        #print("e",self.new_actor.e.grad)
        #print("r",self.new_actor.r.grad)
        #print("w",self.new_actor.W.grad)

        
        return self.actor.T,self.actor.e,self.actor.r,self.actor.W,
    

    
    def train_a(self,memory,p_gamma):
        G,loss=0,0
        for reward,prob in reversed(memory):
            G = reward + p_gamma * G
            loss += -torch.sum(torch.log(prob) * G)
        self.actor_optimizer.zero_grad()

        
        loss.backward()
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f"{name} grad: {param.grad}")
            else:
                print(f"{name} grad is None")
        #print(G)
        #print(loss)
        self.actor_optimizer.step()
        #print(agent_policy.state_dict())
        del loss

        return self.actor.T,self.actor.e,self.actor.r,self.actor.W,

def e_step(agent_num,load_data,T,e,r,w,persona,step,base_time):

    actor = actor = Actor(T,e,r,w,persona)
    print("actorparam",actor.state_dict())
  
    #personaはじめは均等
    policy_ration = torch.empty(step,len(persona[0]),agent_num,agent_num)

    for time in range(step):
        polic_prob = actor.calc_ration(
                    load_data.feature[base_time+time].clone(),
                    load_data.adj[base_time+time].clone(),
                    persona
                    )
     
        policy_ration[time] = polic_prob


    #時間に対して縮約 5,4,32,32 -> 4,32,32
    top = torch.sum(policy_ration,dim = 0)
    #print("top",top.size())

    #分子　全ての時間　 あるペルソナに注目
    rik = torch.empty(agent_num,len(persona[0]))
    #分母 top時間に縮約、bottom:ペルソナについて縮約 4,32,32 -> 32,32
    bottom = torch.sum(top,dim=0)
     
    #print("bo",bottom.size(),bottom)

    # ration 4,32,32
    ration = top/bottom
    #ration = torch.div(top,bottom)
    # ぎょう方向に縮約
    #print("行動単位のration",ration)
    ration = torch.mean(ration,dim=2)
    #print("ration",ration)
    #print("ration",torch.sum(ration,dim=0))

    for i in range(agent_num):
        for k in range(len(persona[0])):
            rik[i,k] = ration[k,i]

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
    persona_ration = torch.from_numpy(persona_ration).to(device)


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
    input_size = 81580

    action_dim = 32
    N = 32
    #パラメータ
    gamma = 0.99
    lamda = 0.95
    lr_a = 0.05
    lr_c = 0.05
    target_update_steps = 8
    alpha = alpha

    beta = beta
    T = np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    )
    e = np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    )

    r = np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    )

    w = np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    )

    persona = persona_ration


    episodes = 32
    story_count = 10
    ln = 0
    ln_sub = 0
    sub_ln = []
    flag = True
    episode = 0
    count = 0
    episodes_reward = []
    persona_parms =[]
    persona_parms.append(torch.from_numpy(np.array([T,e,r,w])))
    persona_rations =[]
    persona_rations.append(persona_ration)


    #n_episodes = 10000

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
            mixture_ratio = e_step(
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
 
        #personaはじめは均等
        if episode == 0:
                    #環境の設定
            obs = Env(
                agent_num=agent_num,
                edges=load_data.adj[LEARNED_TIME].clone(),
                feature=load_data.feature[LEARNED_TIME].clone(),
                temper=T,
                alpha=alpha,
                beta=beta,
                persona=mixture_ratio
            )
    
   
        obs.reset(
                load_data.adj[LEARNED_TIME].clone(),
                load_data.feature[LEARNED_TIME].clone(),
                alpha=alpha,
                beta=beta,
                persona=mixture_ratio
                )
        episode_reward = 0
        trajectory = torch.empty([story_count,2])
        agents = PPO(obs,agent_num, input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,mixture_ratio,story_count)
        memory=[]
        print("pr",mixture_ratio)
        for i in range(story_count):

            edges,feature = obs.state()
            feat,action_probs = agents.get_actions(i,edges,feature)
            #print("0",action_probs[action_probs<0].sum())
            #print("1",action_probs[action_probs>1].sum())
            #print("nan",action_probs[action_probs=="Nan"].sum())
            #->nanいる
            action = action_probs.bernoulli()
            #with torch.no_grad():
            agents.memory.actions[i]=action_probs.clone()
            agents.memory.edges[i] = edges.clone()
            agents.memory.features[i] = feature.clone()
            reward = obs.step(feat,action)
        
            agents.memory.next_edges[i]=action.clone()
            agents.memory.next_features[i]=feat.clone()

            agents.memory.reward[i]=reward.clone()
            episode_reward = episode_reward + reward.sum()
            memory.append((reward.sum(),action_probs))
            #print("ap",action_probs[0])


        episodes_reward.append(episode_reward)
        print("epsiode_rewaerd",episodes_reward[-1])



        new_T,new_e,new_r,new_w = agents.train(gamma,lamda,persona_parms,memory)
        #new_T,new_e,new_r,new_w = agents.train_a(memory,gamma)
        count +=1

        T = new_T
        e = new_e
        r = new_r
        w = new_w
  
     
        persona_parms.append(torch.from_numpy(np.array([T.detach().numpy(),e.detach().numpy(),r.detach().numpy(),w.detach().numpy()])))
        persona_rations.append(mixture_ratio)
        ln_before = ln
        ln = agents.ln
        ln_sub =abs(ln-ln_before)
        episode += 1
        sub_ln.append([ln_sub,episode_reward])
        print("ln_sub---------------------------------",ln_sub)
        episode += 1
        alpha = agents.alpha
        beta = agents.beta

   
        if episode % 10 == 0:
            #print(reward)
            print(episodes_reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")
        if episode >=32:
            flag = False
        #print("T",T,"e",e,"r",r,"w",w,"alpha",alpha,"beta",beta)
    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))
    print(sub_ln)

    for count in range(10):
        obs.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
            alpha=alpha,
            beta=beta,
            persona=persona
        )

        for t in range(TOTAL_TIME - GENERATE_TIME):
            gc.collect()
            #field.state()隣接行列、属性値を返す
            #neighbor_state, feat = field.state()
            #->部分観測(自分のエッジの接続、属性値の情報)にする
          
            edges, feature = obs.state()
            #print("stae",neighbor_state)
            #print("feat",feat)
            feat, action = agents.get_actions(
                t,edges, feature
            )
            del edges, feature

            reward = obs.step(feat,action)

            #予測値
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

            attr_predict_probs = np.concatenate([target_prob], 0)
            try:
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_numpy),
                )
                auc_actv = roc_auc_score(attr_numpy, attr_predict_probs)
            except ValueError as ve:
                print(ve)
                pass
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


            pi_test= agents.memory.test(t)
            #print(len(pi_test))
            #print(len(pi_test[0]))
            #print(len(pi_test[0][0]))
            #flattened_list = [item for sublist1 in pi_test for sublist2 in sublist1 for item in sublist2]
            ##print(len(flattened_list))
            #pi_test = torch.tensor(flattened_list)
            pi_test = pi_test.view(-1)

            gc.collect()
            detach_edge = (
                torch.ravel(load_data.adj[GENERATE_TIME + t])
                .detach()
                .to("cpu")
            )
            #テストデータ
            pos_edge = detach_edge.numpy()
            edge_numpy = np.concatenate([pos_edge], 0)

            #予測データ
            target_prob = pi_test.to("cpu").detach().numpy()

            edge_predict_probs = np.concatenate([target_prob], 0)
    
            #print(target_prob.shape)
            #print(edge_numpy.shape)
            #print(edge_predict_probs.shape)
            # NLLを計算

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