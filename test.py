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
    def __init__(self, agent_num,input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = 81580
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.memory = Memory(agent_num, action_dim)
        self.actor = Actor(T,e,r,w)

        self.critic = Critic(input_size, action_dim)
        #crit
        self.critic_target = Critic(input_size, action_dim)
        #critic.targetをcritic.state_dictから読み込む
        self.critic_target.load_state_dict(self.critic.state_dict())
        #adamにモデルを登録
        self.actors_optimizer = [torch.optim.Adam(self.actor.parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.count = 0

    
    def get_actions(self, edges,feat):
        #観察
        prob,feat,_= self.actor.predict(feat,edges)

        for i in range(self.agent_num): 
            self.memory.pi[i].append(prob[i].tolist())

        action = prob.bernoulli()
        #observation,actionの配列に追加　actions,observationsは要素がエージェントの個数ある配列
        self.memory.observation_edges.append(edges.tolist())
        self.memory.observation_features.append(feat.tolist())
        self.memory.actions.append(action.tolist())
        
        return feat,action

    def train(self):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer
        actions, observation_edges,observation_features, pi, reward= self.memory.get()
        reward = torch.sum(reward,dim=2)
        observation_edges_flatten = torch.tensor(np.array(observation_edges)).view(4,-1)
        observation_features_flatten = torch.tensor(np.array(observation_features)).view(4,-1)
        actions_flatten = torch.tensor(actions).view(4,-1)#4x1024
     

        for i in range(self.agent_num):
            # train actor
            #agentの数input_critic作る

            input_critic = self.build_input_critic(i, observation_edges_flatten,observation_edges, observation_features_flatten, observation_features,actions_flatten,actions)
            Q_target = self.critic_target(input_critic)  
            action_taken = torch.tensor(actions)[:,i,:].type(torch.long).reshape(4, -1)   
            baseline = torch.sum(torch.tensor(pi[i][:]) * Q_target, dim=1)
            max_indices = []
            Q_taken_target = []
        
            for j in range(4):
                Q_taken_target_num = 0
                max_indices.append(torch.where(action_taken[j] == 1)[0].tolist())
                actions_taken = max_indices

                for k in range(len(actions_taken[j])):
                  
                    Q_taken_target_num += Q_target[j][actions_taken[j][k]]

                Q_taken_target.append(Q_taken_target_num)
            Q_taken_target = torch.tensor(Q_taken_target)
            #利得関数
            advantage = Q_taken_target - baseline
            advantage = advantage.reshape(4,1)
            log_pi = torch.log(torch.mul(torch.tensor(pi[i]),action_taken))
            log_pi = torch.where(log_pi == float("-inf"),0,log_pi)
            actor_loss = - torch.mean(advantage * log_pi)
            actor_loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
            #パラメータの更新
            actor_optimizer[i].step()

            # train critic
            Q = self.critic(input_critic)
            Q_taken = []
            for j in range(4):
                Q_value = 0
                for k in range(len(actions_taken[j])):
                
                    Q_value += Q[j][actions_taken[j][k]]
                Q_taken.append(Q_value)
            Q_taken=torch.tensor(Q_taken)

            r = torch.zeros(len(reward[:, i]))
          
            for t in range(len(reward[:, i])):
                #ゴールに到達した場合は,次の状態価値関数は0
                if t == 3:
                    r[t] = reward[:, i][t]
                #ゴールでない場合は、
                else:
                    #Reward + γV(st+1)
                    #print(t, Q_taken_target[t + 1])
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]
        
            #critic_loss = torch.mean((r - Q_taken)**2)
            r = torch.autograd.Variable(r, requires_grad=True)
            Q = torch.autograd.Variable(Q, requires_grad=True)
            critic_loss = torch.mean((r - Q_taken)**2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            critic_optimizer.step()

        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

    def build_input_critic(
            self, agent_id, edges_flatten,
            edges,features_flatten,
            features,actions_flatten,actions
            ):

        batch_size = len(edges)
        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)
        edges_i = torch.tensor([edges[0][agent_id],edges[1][agent_id],edges[2][agent_id],edges[3][agent_id]])
        features_i = torch.tensor([features[0][agent_id],features[1][agent_id],features[2][agent_id],features[3][agent_id]])
        actions_i = torch.empty(1,992)
        #iは時間
        for i in range(4):
            if agent_id == 0:
                action_i = torch.tensor(actions[i][agent_id+1:]).view(1,-1)
            elif agent_id == 32:
                action_i = torch.tensor(actions[i][:agent_id]).view(1,-1)
            else:
                action_i = torch.cat(tensors=(torch.tensor(actions[i][:agent_id]).view(1,-1),torch.tensor(actions[i][agent_id+1:]).view(1,-1)),dim=1)
            
            actions_i = torch.cat(tensors=(actions_i,action_i),dim=0)

        #空の分消す
        actions_i = actions_i[:4]
        input_critic= torch.cat(tensors=(ids,edges_flatten),dim=1)
        input_critic= torch.cat(tensors=(input_critic,actions_i),dim=1)
        input_critic= torch.cat(tensors=(input_critic,features_flatten),dim=1)
        input_critic= torch.cat(tensors=(input_critic,features_i),dim=1)
        input_critic = input_critic.to(torch.float32)

        return input_critic
    

def execute_data():
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
    #alpha,betaの読み込み

    data_size = 32
    persona_num = 4
    LEARNED_TIME = 0
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    load_data = init_real_data()
    agent_num = len(load_data.adj[LEARNED_TIME])
    input_size = 81580
    action_dim = 32
    gamma = 0.99
    lr_a = 0.0001
    lr_c = 0.005
    target_update_steps = 8
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

    T = np.array(
        [0.8 for i in range(persona_num)],
        dtype=np.float32,
    )
    e = np.array(
        [0.8 for i in range(persona_num)],
        dtype=np.float32,
    )

    r = np.array(
        [0.9 for i in range(persona_num)],
        dtype=np.float32,
    )

    w = np.array(
        [1e-2 for i in range(persona_num)],
        dtype=np.float32,
    )

    persona = np.array(
        [0.25 for i in range(4)],
        dtype=np.float32,
    )

    N = len(alpha)

    # E-step
    actor = Actor(T,e,r,w)
#personaはじめは均等

    policy_ration = torch.empty(GENERATE_TIME,len(persona),agent_num,agent_num)
    
    for time in range(GENERATE_TIME):
        polic_prob = actor.calc_ration(
                    load_data.feature[time].clone(),
                    load_data.adj[time].clone(),
                    persona
                    )
        policy_ration[time] = torch.log(polic_prob)
    print(policy_ration.shape)
    for n in range(agent_num):
        persona_rat = policy_ration[:,:,n,:]

         #分子　全ての時間　 あるペルソナに注目
        rik = torch.empty(32,len(persona))
        top = torch.sum(persona_rat,dim = 0)
        print("top",top.shape)
        #print(top)
        #分母 すべての時間,全てのpolicy_ration計算
        bottom = torch.sum(top,dim=0)
        print("bot",bottom.shape)
        #print(bottom)
        #for n in range(len(persona)):
        ration = torch.div(top,bottom)
        print("before",ration)

        for k in range(len(persona)):
            rik[n,k] = ration[k,n]
            
        print(rik)

    print(persona_rat.shape)


    
    #分子　全ての時間　 あるペルソナに注目
    rik = torch.empty(32,len(persona))


    top = torch.sum(policy_ration,dim = 0)
    print(top)
    #分母 すべての時間,全てのpolicy_ration計算
    bottom = torch.sum(top,dim=0)
    print(bottom)
    #for n in range(len(persona)):
    ration = torch.div(top,bottom)
    print("before",rik)
    for i in range(agent_num):
        for k in range(len(persona)):
            rik[i,k] = ration[k,i]
      
    print(rik)

#M-step
    agents = COMA(agent_num, input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w)

    obs = Env(
        edges=load_data.adj[LEARNED_TIME].clone(),
        feature=load_data.feature[LEARNED_TIME].clone(),
        temper=T,
        alpha=alpha,
        beta=beta,
        persona=persona
    )



    episode_reward = 0
    episodes_reward = []

    #n_episodes = 10000
    episodes = 64
    story_count = 4
 
    for episode in range(episodes):

   
        obs.reset(
                load_data.adj[LEARNED_TIME].clone(),
                load_data.feature[LEARNED_TIME].clone()
                )
        episode_reward = 0
        episodes_reward = []
    
        for i in range(story_count):
            #print("start{}".format(i))
            edges,feature = obs.state()
            #print("episode{}".format(episode),"story_count{}".format(i))
            feat,action = agents.get_actions(edges,feature)
            reward = obs.step(feat,action)


            #reward tensor(-39.2147, grad_fn=<SumBackward0>)
            agents.memory.reward.append(reward.tolist())

            episode_reward += reward.sum().item()

          
            #print("end{}".format(i))

          

                #print("done",len(agents.memory.done[0]))
        episodes_reward.append(episode_reward)


       
        #print("train",episode)
        agents.train()

        if episode % 16 == 0:
            #print(reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-16:]) / 16}")

    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    for count in range(10):
        obs.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
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
                edges, feature
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


            pi_test= agents.memory.test()
            #print(len(pi_test))
            #print(len(pi_test[0]))
            #print(len(pi_test[0][0]))
            flattened_list = [item for sublist1 in pi_test for sublist2 in sublist1 for item in sublist2]
            #print(len(flattened_list))
            pi_test = torch.tensor(flattened_list)
            
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
  




if __name__ == "__main__":
    execute_data()