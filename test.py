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
from critic import TimeSeriesGCN
from critc import Critic
from beforecritic import CriticBefore

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
        self.new_actor = Actor(T,e,r,w,rik)
        #self.critic = TimeSeriesGCN(self.input_size,story_count)
        self.critic = Critic(T,e,r,w,rik)
        self.new_critic = Critic(T,e,r,w,rik)
        #self.new_critic = TimeSeriesGCN(self.input_size,story_count)

        #adamにモデルを登録
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a) 
        self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr_a) 
        self.new_critic_optimizer = torch.optim.Adam(self.new_critic.parameters(), lr=lr_c) 
        #self.new_critic_optimizer = torch.optim.Adam(self.new_critic.parameters(), lr=lr_c) 




    def get_actions(self,env, edges,feat):
        
        with torch.no_grad():
        
            prob,feat = self.actor.predict(feat,edges)
    
        return feat,prob
    

    
    def train(self,gamma,lamda,param):

        #----gae----
        G,loss = 0,0
        cliprange=0.2
        storycount = self.story_count


       
        #print("---")
        #print(pi)
        #print(old_parm)
        #print(mixture)

        lnpxz = self.memory.actions.view(-1).sum()
        v = torch.empty([storycount,32,32])
        n_v = torch.empty([storycount,32,32])
        for i in range(len(self.memory.features)):
            with torch.no_grad():
                v[i] = self.critic.predict(self.memory.features[i],self.memory.edges[i]).clone()
                #v = self.critic.predict(self.memory.edges,self.memory.features)
            #n_v = self.new_critic.predict(self.memory.next_edges,self.memory.next_features)
            n_v[i] = self.new_critic.predict(self.memory.features[i],self.memory.edges[i]).clone()

        gae_li = torch.empty(storycount,32,32)
        gae = 0
        #print(n_v.size()) 10x1
        #print("aa",self.memory.reward.size()) 10x32x1
        #slf.,memory.reward 10x32x1
        #r1x32x1
        for i in reversed(range(len(self.memory.reward))):
            r = self.memory.reward[i]
            print(v[i])
            #print("r",r.size())#10x32x1
            #print("v",v.size())#10x32x32

            if i == len(self.memory.reward) - 1:
                    # エピソード最後は次の状態価値はない
                delta = r - v[i]
                
        
            else:
                    # delta = r + γV(t+1) - V(t)
                delta = r + gamma * n_v[i] - v[i]
            print("r",r)
            print("gae",gae)
                
                # A = delta(t) + γ*λ*delta(t+1)
            gae = delta +  gamma * lamda * gae
            #gae 32x32
            #print(gae)
                # 各stepのGAEを保存し、バッチとする
            gae_li[i] =gae
                #batch = self.recent_batch[i]
                #batch["discounted_reward"] = gae
                #self.remote_memory.add(batch)
  

        #-----------
        losses = []
      
        
        v_targs = gae_li + n_v

        #print(v_targs.size())→10x32x1
        for i in range(storycount):
            #v = self.critic.predict(self.memory.edges,self.memory.features)
            #n_v = self.critic.predict(self.memory.next_edges,self.memory.next_features)
            old_vpred = v[i]
            vpred = n_v[i]

            self.new_critic_optimizer.zero_grad()

            vpred_clipped = old_vpred + torch.clamp(vpred - old_vpred, min=-cliprange, max=cliprange)

            # 損失の計算
            loss1 = (v_targs[i] - vpred).pow(2)
            loss2 = (v_targs[i] - vpred_clipped).pow(2)
            loss = torch.max(loss1, loss2).mean()

            # パラメータの更新
            self.new_critic_optimizer.step()

            loss.backward(retain_graph=True)
            

            # 勾配のクリッピング
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            losses.append(loss)
            #print("optim",self.critic.parameters())

            #for name, param in self.critic.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name} grad: {param.grad}")
            #    else:
            #       print(f"{name} grad is None")
            #print("done critic")
        state_dict = self.new_critic.state_dict()
            #self.critic = TimeSeriesGCN(self.input_size)
            # モデルBに state_dict をロード
        self.critic.load_state_dict(state_dict)
        
            #print("critic",loss.grad)

            #with tf.GradientTape() as tape:

            #   vpred = n_v[i]
            #    vpred_clipped = old_vpred + torch.clamp(
            #        vpred - old_vpred, min=-cliprange, max=cliprange)

            #    loss = tf.maximum(
            #        tf.square(tf.convert_to_tensor((v_targs[i] - vpred).detach().numpy())),
            #        tf.square(tf.convert_to_tensor((v_targs[i] - vpred_clipped).detach().numpy())))

            #    loss = tf.reduce_mean(loss)

            #grads = tape.gradient(loss, self.critc.trainable_variables)
            #grads, _ = tf.clip_by_global_norm(grads, 0.5)
            #self.critic.optimizer.apply_gradients(
            #    zip(grads, self.critic.trainable_variables))

            #losses.append(loss)

            #np.array(losses).mean() 

        
        
        for i in range(storycount):
            old_policy = self.memory.actions[i]
            self.actor_optimizer.zero_grad()
            #gae_t = batche_li[]

            # 新しいポリシーの計算
            new_policy,_ =  self.new_actor.forward(self.memory.features[i],self.memory.edges[i])

            # old/newの比率の計算
            #print("new",new_policy[0])
            #print("dict",self.new_actor.state_dict())

            #print(old_policy[0])
            #print("dict",self.actor.state_dict())
   
            ratio =torch.exp(torch.log(new_policy+1e-7) - torch.log(old_policy+1e-7))
            #ratio = torch.where(torch.isnan(ratio), torch.full_like(ratio, 1.0), ratio)

            # クリッピング
            ratio_clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

            # 損失の計算
            loss_unclipped = ratio * gae_li[i]
            loss_clipped = ratio_clipped * gae_li[i]
        
            #print("GAE",gae_li.size())
            #print("ratio",ratio)
            #print("ratio",gae_li[i])
            #print("loss_clip",loss_clipped)
            # 最小値を取る
            loss = torch.min(loss_unclipped, loss_clipped)
            
            # 最大化のために-1を掛ける
            loss = -loss.mean()
            #print("loss",loss)
          
            # 勾配の計算と適用
            loss.backward(retain_graph=True)
            self.new_actor_optimizer.step()

           

            #with tf.GradientTape() as tape:
            #    new_policy = self.actor(self.memory.edges[i],self.memory.features[i])
                #old/new
            #    ratio = tf.exp(np.log(old_policy)-np.log(new_policy))
            #    ratio_clipped = torch.clamp(
            #        ratio, min=1 - cliprange, max=1 + cliprange)

            #    loss_unclipped = ratio * gae[i]
            #    loss_clipped = ratio_clipped * gae[i]
                    
                #: clipされていない代理目的関数とclipped代理目的関数の小さい方を採用
            #    loss = tf.minimum(loss_unclipped, loss_clipped)
                    
                #: 最大化したいので-1を掛ける
            #    loss = -1 * tf.reduce_mean(loss)


            #grads = tape.gradient(loss, self.policy.trainable_variables)
            #self.policy.optimizer.apply_gradients(
            #  zip(grads, self.policy.trainable_variables)) """
           

        #with torch.no_grad():
           
        #    self.actor.T -= 0.01 * self.actor.T.grad
       #     self.actor.e -= 10000 * self.actor.e.grad
       #     self.actor.r -= 0.01 * self.actor.r.grad
       #     self.actor.W -= 0.01 * self.actor.W.grad

   
        state_dict = self.new_actor.state_dict()
        # モデルBに state_dict をロード
        self.actor.load_state_dict(state_dict)
        
        print("critic",loss.grad)

        return self.actor.T,self.actor.e,self.actor.r,self.actor.W,self.alpha,self.beta
    

    

def e_step(agent_num,load_data,T,e,r,w,persona,step,base_time):

    actor = actor = Actor(T,e,r,w,persona)
    print("actorparam",actor.state_dict())
  
    #personaはじめは均等
    policy_ration = torch.empty(step,len(persona[0]),agent_num)

    for time in range(step):
        polic_prob = actor.calc_ration(
                    load_data.feature[base_time+time].clone(),
                    load_data.adj[base_time+time].clone(),
                    persona
                    )
  
        policy_ration[time] = polic_prob


    #分子　全ての時間　 あるペルソナに注目
    rik = torch.empty(agent_num,len(persona[0]))


    top = torch.sum(policy_ration,dim = 0)

    #分母 すべての時間,全てのpolicy_ration計算
    bottom = torch.sum(top,dim=0)

    #for n in range(len(persona)):
    ration = torch.div(top,bottom)

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
    lr_a = 0.01
    lr_c = 0.01
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
        [0.8 for i in range(persona_num)],
        dtype=np.float32,
    )

    w = np.array(
        [0.8 for i in range(persona_num)],
        dtype=np.float32,
    )

    persona = persona_ration


    episodes = 32
    story_count = 32
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

        episodes_reward.append(episode_reward)



        agents.train(gamma,lamda,persona_parms)
        count +=1

        T = agents.new_actor.T
        e = agents.new_actor.e
        r = agents.new_actor.r.squeeze(1)
        w = agents.new_actor.W.squeeze(1)
  
     
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

        print("pr",mixture_ratio)
        if episode % 10 == 0:
            #print(reward)
            print(episodes_reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")
        if episode >= 2:
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