# Standard Library
import gc

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score




# First Party Library
import csv
import config
from env import Env
from init_real_data import init_real_data

#another file
from memory import Memory
from actor import Actor
from critic import Critic

device = config.select_device

class COMA:
    def __init__(self, obs,agent_num,input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,rik):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = 81580
        self.gamma = gamma
        self.persona = rik
        self.target_update_steps = target_update_steps
        self.memory = Memory(agent_num, action_dim)
        #self.actor = [Actor(T[i],e[i],r[i],w[i],rik[i]) for i in range(len(alpha))]
        self.actor = Actor(T,e,r,w,rik)
        self.critic = Critic(input_size, action_dim)
        self.obs = obs
        self.alpha = self.obs.alpha
        self.beta = self.obs.beta
        #crit
        self.critic_target = Critic(input_size, action_dim)
        #critic.targetをcritic.state_dictから読み込む
        self.critic_target.load_state_dict(self.critic.state_dict())
        params = [self.obs.alpha,self.obs.beta]
        #adamにモデルを登録
        #self.actors_optimizer = [torch.optim.Adam(self.actor.parameters(), lr=lr_a) for i in range(agent_num)]
        self.actors_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a) 
        self.critic_optimizer = torch.optim.Adam(params, lr=0.01)
        self.ln = 0
        self.count = 0

    
    def get_actions(self,time, edges,feat):
        #観察
        #tenosr返す
        #norm = feat.norm(dim=1)[:, None] 
        #norm = norm + 1e-8
        #feat = feat.div(norm)
        prob,feat,_= self.actor.predict(feat,edges)
    
        #print("prob",prob.shape)
        for i in range(self.agent_num): 
            self.memory.pi[time][i]=prob[i]

        action = prob.bernoulli()
        #observation,actionの配列に追加　actions,observationsは要素がエージェントの個数ある配列
        self.memory.observation_edges[time]=edges
        self.memory.observation_features[time]=feat
        self.memory.actions[time]=action


        
        return feat,action

    def train(self):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer
        actions, observation_edges,observation_features, pi, reward= self.memory.get()
        lnpxz = pi.view(-1).sum()
        print("ln",lnpxz)
        self.ln = lnpxz
        #print("REWL75",reward[0])
        #reward = torch.sum(reward,dim=2)
        #viewは5->10までの区間を予測したいので5
        #observation_edges_flatten = torch.tensor(observation_edges).view(5,-1)
        observation_edges_flatten = observation_edges.view(5,-1)
        #observation_features_flatten = torch.tensor(observation_features).view(5,-1)
        observation_features_flatten = observation_features.view(5,-1)
        #actions_flatten = torch.tensor(actions).view(5,-1)#4x1024
        actions_flatten = actions.view(5,-1)#4x1024
        #実験用後から消す
        actor_loss = 0
        critic_loss = 0
        for i in range(self.agent_num):
            # train actor
            #agentの数input_critic作る

            input_critic = self.build_input_critic(i, observation_edges_flatten,observation_edges, observation_features_flatten, observation_features,actions_flatten,actions)
            Q_target = self.critic_target(input_critic)  
            action_taken = actions[:,i,:].type(torch.long).reshape(5, -1)  

            #COMA 反証的ベースライン
            baseline = torch.sum(pi[i,:] * Q_target, dim=1)
            max_indices = []
            Q_taken_target =torch.empty(5)
            #Q_taken_li =[]

            #5->10を予測するので5
            for j in range(5):
                Q_taken_target_num = 0
                max_indices.append(torch.where(action_taken[j] == 1)[0].tolist())
                actions_taken = max_indices

                for k in range(len(actions_taken[j])):
                  
                    Q_taken_target_num =Q_taken_target_num + Q_target[j][actions_taken[j][k]]
                    #Q_taken_li.append(Q_taken_target_num + Q_target[j][actions_taken[j][k]])

                Q_taken_target[j] = Q_taken_target_num
                #Q_taken_target = torch.tensor(Q_taken_li.sum())
            #print(max_indices)
            
            #利得関数
            advantage = Q_taken_target - baseline
            #print("ad",advantage.shape)
            advantage_re = advantage.reshape(5,1)
            log_pi_mul = torch.mul(pi[i],action_taken) 
            #print("lpm",log_pi_mul[0])
            log_pi_mul[log_pi_mul == float(0.0)] = torch.exp(torch.tensor(2.0))
            #print(action_taken.shape)
            #print(torch.tensor(pi[i]).shape)
            # loge:torch.log(torch.exp(torch.tensor(1.0)))
            #print("lpm",log_pi_mul[0])
            log_pi = torch.log(log_pi_mul)
            #print("lp",log_pi[0])

            log_pi[log_pi == float(2.0)] = 0.0
            #print("lp",log_pi[0])

            #log_pi = torch.where(log_pi_mul == float("-inf"),0,log_pi_mul)
            # #通常は、計算グラフから分離されないが-infあるので分離される
        
            #print(log_pi[0])
            #print(log_pis[0])
            #以下のように変更
            #log_pis[log_pis == float("-inf")] = 0.0
            #print(log_pis[0])
            # = - を +=に変更
            actor_loss =actor_loss - torch.mean(advantage_re * log_pi)
            #print("loss",actor_loss)
            #actor_optimizer.zero_grad() 
            #actor_loss.backward(retain_graph=True)
            
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
            #パラメータの更新
            #actor_optimizer.step()
   
            # パラメータがrequires_grad=Trueになっているか確認
            #for name, param in self.actor.named_parameters():
            #    print(name, param.requires_grad)
           
            #print("param",self.actor.state_dict())

            # train critic
            Q = self.critic(input_critic)
            Q_taken = torch.empty(5)
            for j in range(5):
                Q_value = 0
                for k in range(len(actions_taken[j])):
                
                    Q_value =Q_value + Q[j][actions_taken[j][k]]
                Q_taken[j]=Q_value
      
            r = torch.empty(len(reward[:, i]))

            for t in range(len(reward[:, i])):
                #ゴールに到達した場合は,次の状態価値関数は0
                if t == 4:
                    r[t] = reward[:, i][t]
                    #r_li.append(reward[:, i][t])
                #ゴールでない場合は、
                else:
                    #Reward + γV(st+1)
                    #print(t, Q_taken_target[t + 1])
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]



            critic_loss = critic_loss + torch.mean((r - Q_taken)**2)
           
        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        print("a&b_grad",self.obs.alpha,self.obs.beta)

        #critic_optimizer.step()
       
        with torch.no_grad():
            self.obs.alpha -= 0.001 * self.obs.alpha.grad
            self.obs.beta -= 0.001 * self.obs.beta.grad

        print("a&b_grad",self.obs.alpha.grad,self.obs.beta.grad,self.obs.persona.grad)
        print("a&b_grad",self.obs.alpha,self.obs.beta)
        print("a&b_grad",self.obs.alpha.grad.is_leaf,self.obs.beta.grad.is_leaf)
   
     
        #actor
            #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)



        actor_optimizer.zero_grad() 
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        #パラメータの更

        #actor_optimizer.step()
        print("log_pi_grad",log_pi.grad_fn)
        print("log_pi_mul_grad",log_pi_mul.grad_fn)
        print("advantage_grad",advantage.grad_fn)
        print("pi_grad",pi.grad_fn)
        print("pi_grad",self.memory.pi.grad_fn)
        print("T_grad",self.actor.T.grad)
        print("e_grad",self.actor.e.grad)
        print("r_grad",self.actor.r.grad)
        print("w_grad",self.actor.W.grad)
        with torch.no_grad():
            self.actor.T -= 0.001 * self.actor.T.grad
            self.actor.e -= 0.001 * self.actor.e.grad
            self.actor.r -= 0.001 * self.actor.r.grad
            self.actor.W -= 0.001 * self.actor.W.grad
            
          



        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

        return self.actor.T,self.actor.e,self.actor.r,self.actor.W,self.alpha,self.beta
    


    def build_input_critic(
            self, agent_id, edges_flatten,
            edges,features_flatten,
            features,actions_flatten,actions
            ):

        batch_size = len(edges)
        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)
        #5->10のデータの整形
        edges_i = torch.empty(5,32)
        edges_i = edges[:,agent_id]
        features_i = torch.empty(5,features.shape[2])
        features_i = features[:,agent_id]
        actions_i = torch.zeros(1,992)
        
        #iは時間
        #5->10までの区間を予測したいので5
        for i in range(5):
            if agent_id == 0:
                action_i = actions[i,agent_id+1:].view(1,-1)
            elif agent_id == 32:
                action_i = actions[i,:agent_id].view(1,-1)
            else:
                action_i = torch.cat(tensors=(actions[i,:agent_id].view(1,-1),actions[i,agent_id+1:].view(1,-1)),dim=1)
            
            actions_i = torch.cat(tensors=(actions_i,action_i),dim=0)

        #空の分消す
        #print(actions_i.shape)
        #print(actions_i[0])
        actions_i = actions_i[1:]
        #print(actions_i[0])
        input_critic= torch.cat(tensors=(ids,edges_flatten),dim=1)
        input_critic= torch.cat(tensors=(input_critic,actions_i),dim=1)
        input_critic= torch.cat(tensors=(input_critic,features_flatten),dim=1)
        input_critic= torch.cat(tensors=(input_critic,features_i),dim=1)
        input_critic = input_critic.to(torch.float32)

        return input_critic
    


#EMのEstep 全てtenosr返す
def e_step(agent_num,load_data,T,e,r,w,persona,step,base_time):

    actor = Actor(T,e,r,w,persona)
  
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
    rik = torch.empty(32,len(persona[0]))


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


    #初期の設定(データサイズ、ペルソナの数、学習、生成時間)
    data_size = 32
    persona_num = 8
    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    #データの読み込み
    load_data = init_real_data()
    agent_num = len(load_data.adj[LEARNED_TIME])
    #パラメータの設定
    input_size = 81580
    action_dim = 32
    gamma = 0.99
    lr_a = 0.1
    lr_c = 0.01
    target_update_steps = 8

    data_persona = []
    #ペルソナ2の時
    #--------------
    path = "/Users/matsumoto-hirotomo/coma/data_norm{}.csv".format(int(persona_num))
    csvfile = open(path, 'r')
    gotdata = csv.reader(csvfile)
    for row in gotdata:
        data_persona.append(int(row[2]))
    csvfile.close()
    
    alpha = torch.from_numpy(
        np.array(
            [1.0 for i in range(persona_num)],
            dtype=np.float32,
        ),
    ).to(device)

    beta = torch.from_numpy(
        np.array(
            [1.0 for i in range(persona_num)],
            dtype=np.float32,
        ),
    ).to(device)

    T = torch.from_numpy(
        np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    ),
    ).to(device)
    e = torch.from_numpy(
        np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    ),
    ).to(device)

    r = torch.from_numpy(
        np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    ),
    ).to(device)

    w = torch.from_numpy(
        np.array(
        [1.0 for i in range(persona_num)],
        dtype=np.float32,
    ),
    ).to(device)

    persona = torch.from_numpy(
        np.array(
        [0.25 for i in range(4)],
        dtype=np.float32,
    ),
    ).to(device)

    N = len(alpha)



    #n_episodes = 10000
    episodes = 100
    episode = 0
    story_count = 5
    episodes_reward = []
    ln_sub = 100
    ln = 500

    #while ln_sub > 0.1:
    for i in range(episodes):

        # E-step
        if i == 0:
            rik = torch.empty(32,persona_num)
            for i in range(agent_num):
                    rik[i] = (1-0.3)/persona_num
                    rik[i][data_persona[i]] = 0.3
            mixture_ratio = rik
            persona_ration = rik
            print(mixture_ratio.size())

            

         
        else:
            print(i)
            print("-------------------------")
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
            print("mm",mixture_ratio.size())
        #personaはじめは均等
        if episode == 0:
                    #環境の設定
            obs = Env(
                agent_num = agent_num,
                edges=load_data.adj[LEARNED_TIME].clone(),
                feature=load_data.feature[LEARNED_TIME].clone(),
                alpha=alpha,
                beta=beta,
                persona=mixture_ratio
            )
  


        print(mixture_ratio)
        #M-step
        obs.reset(
                load_data.adj[LEARNED_TIME].clone(),
                load_data.feature[LEARNED_TIME].clone(),
                alpha=alpha,
                beta=beta,
                persona=mixture_ratio
                )
        #print("T",T,e,r,w)
        #print("mixture_ration",mixture_ratio[0])
        
        episode_reward = 0
        
        agents = COMA(obs,agent_num, input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,mixture_ratio)


        #n_episodes = 10000

    



        #いらんくねepisodes_reward = []
    
        for i in range(story_count):
            #print("start{}".format(i))
            edges,feature = obs.state()
            #print("episode{}".format(episode),"story_count{}".format(i))
            feat,action = agents.get_actions(i,edges,feature)
            reward = obs.step(feat,action)


            #reward tensor(-39.2147, grad_fn=<SumBackward0>)
            agents.memory.reward[i]=reward
            #print("reward",agents.memory.reward[0])

            episode_reward += reward.sum()

          
            #print("end{}".format(i))

          

                #print("done",len(agents.memory.done[0]))
        episodes_reward.append(episode_reward)


       
        #print("train",episode)
        agents.train()

        T = agents.actor.T
        e = agents.actor.e
        r = agents.actor.r
        w = agents.actor.W
       
        alpha = agents.alpha
        beta = agents.beta
        print("alpha",alpha)
        ln_before = ln
        ln = agents.ln
        ln_sub =abs(ln-ln_before)
        episode += 1
        print("ln_sub",ln_sub)
        print("T",T,"e",e,"r",r,"w",w,"alpha",alpha,"beta",beta)
        

        if episode % 10 == 0:
            #print(reward)
            print("T",T,"e",e,"r",r,"w",w,"alpha",alpha,"beta",beta)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-16:]) / 16}")

    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    for count in range(10):
        obs.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
            alpha=alpha,
            beta=beta,
            persona=mixture_ratio
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

            target_prob = torch.ravel(feat).to("cpu")
         
            gc.collect()
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
    

    np.save("proposed_edge_auc", calc_log)
    np.save("proposed_edge_nll", calc_nll_log)
    np.save("proposed_attr_auc", attr_calc_log)
    np.save("proposed_attr_nll", attr_calc_nll_log)
    np.save("parameter",np.concatenate([alpha.detach(),beta.detach().numpy(),T.detach().numpy(),e.detach().numpy()],axis=0))
    np.save("rw_paramerter",np.concatenate([r.detach().numpy().reshape(1,-1),w.detach().numpy().reshape(1,-1)],axis=0))




if __name__ == "__main__":
    execute_data()