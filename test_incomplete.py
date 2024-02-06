# Standard Library
import gc

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import optuna



# First Party Library
import csv
import config
from env import Env
from init_real_data import init_real_data

#another file
from memory import Memory
from actor import Actor
from critic import Critic
from beforecritic import CriticBefore

device = config.select_device

class COMA:
    def __init__(self, obs,agent_num,input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,rik,story_count):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = 81580
        self.gamma = gamma
        self.persona = rik
        self.target_update_steps = target_update_steps
        self.story_count = story_count 
        self.memory = Memory(agent_num, action_dim,self.story_count)
        #self.actor = [Actor(T[i],e[i],r[i],w[i],rik[i]) for i in range(len(alpha))]
        self.obs = obs
        self.actor = Actor(T,e,r,w,rik)
        num_features = self.obs.feature.size()[1]
        #t-1での単位行列のサイズ
        before_features = self.obs.edges.size()[1]
        print(before_features)
        self.critic = Critic(num_features, num_features//10, 1)
        self.critic_before = CriticBefore(before_features,before_features//10, 1)

        self.alpha = self.obs.alpha
        self.beta = self.obs.beta
        #crit

        self.critic_target = Critic(num_features,  num_features//10, 1)
        self.critic_target_before = CriticBefore(before_features, before_features//10, 1)
        #critic.targetをcritic.state_dictから読み込む
        self.critic_target.load_state_dict(self.critic.state_dict())
        params = [self.obs.alpha,self.obs.beta]
        #adamにモデルを登録
        #self.actors_optimizer = [torch.optim.Adam(self.actor.parameters(), lr=lr_a) for i in range(agent_num)]
        self.actors_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a) 
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01, weight_decay=5e-4)
        self.critic_before_optimizer = torch.optim.Adam(self.critic_before.parameters(), lr=0.01, weight_decay=5e-4)
        self.ln = 0

        self.count = 0


    def get_actions(self,time, edges,feat):
  
        prob,feat,_= self.actor.predict(feat,edges)
    
        #print("prob",prob.shape)
        for i in range(self.agent_num): 
            self.memory.pi[time][i]=prob[i]
            #print(prob[i])
            

        action = prob.bernoulli()
        #observation,actionの配列に追加　actions,observationsは要素がエージェントの個数ある配列
        self.memory.observation_edges[time]=edges
        self.memory.observation_features[time]=feat
        self.memory.actions[time]=action

        return feat,action
    
    def create_edge_features(self,node_features, edge_index,i):
    # エッジのノード特徴を取得
        #
        #反証行動の中から特定のagentに関する接続情報を取得
        agent_edge_index = (edge_index==i).nonzero(as_tuple=False).t().contiguous()
        #print(edge_index[0][edge_index[0]==i])
        #print(edge_index[1][edge_index[0]==i])
        #一つ目のtenosr 0は元のtenosr1行目(to) 1は2行目(go) 行番号
        #2つ目のtenosr 列番号
        #node_featuresは３２なのでedge_indexでリンク元リンク先の特徴をとる
        #edge_indexは0起点で相手がわかるので
        #print(edge_index[0][edge_index[0]==i])

        #edge_features = node_features[agent_edge_index[0][agent_edge_index[0]==0]] + node_features[edge_index[1][edge_index[0]==i]]
        edge_features = node_features[edge_index[0][edge_index[0]==i]] + node_features[edge_index[1][edge_index[0]==i]]
        #edge_features = torch.cat((agent_edge_index[0][agent_edge_index[0]==0].view(-1,1),edge_index[1][edge_index[0]==i].view(-1,1),edge_features),dim=1)
        #これでも良いがPiが32,20,32とかいう訳のわからん形状
        ########
        #tensor_edge_features = torch.zeros(32,32)
        #tensor_edge_features.index_put_((agent_edge_index[0][agent_edge_index[0]==0],edge_index[1][edge_index[0]==i]),edge_features.view(1,-1).squeeze())
        ########
        tensor_edge_features = torch.zeros(32)
        input_edge_feature = edge_features.detach().clone().view(1,-1).squeeze()
        tensor_edge_features[edge_index[1][edge_index[0]==i]]= input_edge_feature

        return tensor_edge_features
    
    def create_taken_edge_features(self,node_features, edge_index,i):
    # エッジのノード特徴を取得
     
        #agent_edge_index = (edge_index==i).nonzero(as_tuple=False).t().contiguous()
        

        
        edge_features_taken = node_features[edge_index[0][edge_index[0]==i]] + node_features[edge_index[1][edge_index[0]==i]]
        #print(edge_features_taken)
        tensor_edge_features_taken = torch.zeros(32)
        input_edge_feature_taken = edge_features_taken.detach().clone().view(1,-1).squeeze()
        tensor_edge_features_taken[edge_index[1][edge_index[0]==i]]= input_edge_feature_taken
        #print(tensor_edge_features_taken)
        return tensor_edge_features_taken

    
    def train(self):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer
        critic_before_optimizer = self.critic_before_optimizer
        actions, observation_edges,observation_features, pi, reward= self.memory.get()

        lnpxz = pi.view(-1).sum()
        #print("ln",lnpxz)
        self.ln = lnpxz
       
        actor_loss = 0
        critic_loss = 0
        
        self.critic_target.train() 
        self.critic.train() 
        #Q値を保管するリスト 
        #このtensorには実際に行っていない行動に対しての評価値が欲しい
        #つまり,エッジの回帰？
        real_Q_target = torch.empty(32,self.story_count,32)
        real_Q_taken = torch.empty(32,self.story_count,32)

        #print("エッジ",(observation_edges[0]==0).nonzero(as_tuple=False).t().contiguous()[0])
        for i in range(self.agent_num):
            #一つ前の情報を使うので-1->なし
            
            for t in range(self.story_count):
                #反証行動
                #反証行動のみ,[32,32]要素0から置き換える
                #critic_targetは,エッジ生成の評価値が欲しい
                input_critic = (observation_edges[t]==1).nonzero(as_tuple=False).t().contiguous()
                input_crtic_target = (observation_edges[t]==0).nonzero(as_tuple=False).t().contiguous()
                Q = self.critic(input_critic,observation_features[t])
                Q_target = self.critic_target(input_crtic_target,observation_features[t])
                #Q_target = Q_target.squeeze() 
                
                #反証行動
                edge_index = (observation_edges[t]==0).nonzero(as_tuple=False).t().contiguous()
                edge_features = self.create_edge_features(Q_target,edge_index,i)

                #実際に取った行動
                edge_index_taken = (observation_edges[t]==1).nonzero(as_tuple=False).t().contiguous()
                edge_features_taken = self.create_taken_edge_features(Q,edge_index_taken,i)
                
                #Q_beforeいらないかも〜
                #Q_target_before = self.critic_target_before(observation_edges[t].nonzero(as_tuple=False).t().contiguous(),self.obs.identity) 
                #real_Q_target[t]  = (Q_target + Q_target_before).squeeze()
    
                real_Q_target[i,t,:]  = edge_features
                real_Q_taken[i,t,:] = edge_features_taken

        #反証的ベースライン
        #これで32x20x32を返す
        #baseline = pi * real_Q_target
        #→時間の和をとる
        #32x32で各行はagentaの反証行動の値
        baseline = torch.sum(pi * real_Q_target,dim=1)
        max_indices = []
        Q_taken_target =torch.empty(self.story_count,self.agent_num)
            #Q_taken_li =[]
        for i in range(self.agent_num):
            action_taken = actions[:,i,:].type(torch.long).reshape(self.story_count, -1)  
            #5->10を予測するので5
            for j in range(self.story_count):
                Q_taken_target_num = 0

                max_indices.append(torch.where(action_taken[j] == 1)[0].tolist())
                actions_taken = max_indices
                Q_taken_target[j] = real_Q_taken[i][j]*action_taken[j]

            #利得関数
            advantage = Q_taken_target - baseline[i]
            #あるagent:iのすべての時間での他のノードととのエッジの生成確率
            #advantage_re = advantage.reshape(self.story_count,1)
            advantage_re = advantage
            log_pi_mul = torch.mul(pi[i],action_taken) 
            log_pi_mul[log_pi_mul == float(0.0)] = torch.exp(torch.tensor(2.0))
            log_pi = torch.log(log_pi_mul)
            log_pi[log_pi == float(2.0)] = 0.0
            actor_loss =actor_loss - torch.mean(advantage_re * log_pi)
            r = torch.empty(len(reward[:, i]))

            for t in range(len(reward[:, i])):
                    #ゴールに到達した場合は,次の状態価値関数は0
                if t == self.story_count-1:
                    r[t] = reward[:, i][t]
                        #r_li.append(reward[:, i][t])
                    #ゴールでない場合は、
                else:
                        #Reward + γV(st+1)
                        #print(t, Q_taken_target[t + 1])
                    #reward[:, i][t]はスカラー
                    r[t] = reward[:, i][t] + self.gamma * torch.sum(real_Q_taken[i][t + 1])



            critic_loss = critic_loss + torch.mean((r - torch.sum(real_Q_taken[i]))**2)
        # train critic
        print("i reach here")
        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        #print("a&b_grad",self.obs.alpha,self.obs.beta)

        critic_optimizer.step()
       

 
        #actor
            #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)



        actor_optimizer.zero_grad() 
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)



        with torch.no_grad():
           
            self.actor.T -= 1 * self.actor.T.grad
            self.actor.e -= 10000 * self.actor.e.grad
            self.actor.r -= 0.08 * self.actor.r.grad
            self.actor.W -= 0.08 * self.actor.W.grad

        print("T_grad",self.actor.T.grad,str(0.0001 * self.actor.T.grad),self.actor.T)
        print("e_grad",self.actor.e.grad,str(10000 * self.actor.e.grad),self.actor.e)
        print("r_grad",self.actor.r.grad,str(0.0001 * self.actor.r.grad),self.actor.r)
        print("w_grad",self.actor.W.grad,str(10 * self.actor.W.grad),self.actor.W)
            




        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

        return self.actor.T,self.actor.e,self.actor.r,self.actor.W,self.alpha,self.beta
    

    

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


def execute_data(skiptime,drop):

    #ペルソナ数の設定:ペルソナの数[3,4,5,6,8,12]
    persona_num = 5
    path = "experiment_data/NIPS/200_20/incomplete/t={}/drop={}/persona={}/gamma{}.npy".format(skiptime,drop,int(persona_num),int(persona_num))
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    persona_ration = torch.from_numpy(persona_ration).to(device)


    #テストデータの入力
    LEARNED_TIME = 4
    #生成したい時間の範囲
    GENERATE_TIME = 5
    #全体の時間の範囲
    TOTAL_TIME = 10

    #データのロード
    load_data = init_real_data()
    #dropoutしたノードi

    #あるノードにi関する情報を取り除く
    #list[tensor]のキモい構造なので
    load_data.adj[4][drop,:] = 0
    load_data.adj[4][:,drop] = 0
    #load_data.feature[4][i][:] = 0
    path = "experiment_data/NIPS/200_20/incomplete/t={}/drop={}/persona={}/means{}.npy".format(skiptime,drop,int(persona_num),int(persona_num))
    means = np.load(path)
    means = means.astype("float32")
    #学習についての諸設定
    agent_num = len(load_data.adj[LEARNED_TIME])
    input_size = 81580
    action_dim = 32
    


    #パラメータの初期値の設定
    means = torch.from_numpy(means).to(device)
    alpha = means[:,0]
    beta = means[:,1]
    gamma = 0.90
    lr_a = 0.1
    lr_c = 0.005
    target_update_steps = 10
 
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

    N = len(alpha)

    episodes = 32
    story_count = 20
    ln = 0
    sub_ln = []
    flag = True
    episode = 0




    #n_episodes = 10000

    while flag or ln_sub <= 1:
        


        # E-step
        if episode == 0:

            mixture_ratio = persona_ration
       
        else:
            print(episode)
            print("-------------------------")
            mixture_ratio = persona_ration
 
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
        episodes_reward = []
        agents = COMA(obs,agent_num, input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w,mixture_ratio,story_count)



        for i in range(story_count):

            edges,feature = obs.state()
            feat,action = agents.get_actions(i,edges,feature)
            reward = obs.step(feat,action)
            agents.memory.reward[i]=reward
            episode_reward += reward.sum()

        episodes_reward.append(episode_reward)


        agents.train()
        T = agents.actor.T
        e = agents.actor.e
        r = agents.actor.r
        w = agents.actor.W
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
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")
            print("T",T,"e",e,"r",r,"w",w,"alpha",alpha,"beta",beta)
        if episode >= 100:
            flag = False
    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))
    print(sub_ln)
    agents.critic_target.train() 
    agents.critic.train() 
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
            #NLL
            # NLLを計算
            criterion = nn.CrossEntropyLoss()
            error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_numpy),
                )
            auc_actv = roc_auc_score(attr_numpy, attr_predict_probs)
       
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
    print(edge_predict_probs)

   
    spath= "/Users/matsumoto-hirotomo/coma/experiment_data/NIPS/200_20/incomplete/t={}/drop={}/".format(skiptime,drop)

    np.save(spath+"persona={}/proposed_edge_auc".format(persona_num), calc_log)
    np.save(spath+"persona={}/proposed_edge_nll".format(persona_num), calc_nll_log)
    np.save(spath+"persona={}/proposed_attr_auc".format(persona_num), attr_calc_log)
    np.save(spath+"persona={}/proposed_attr_nll".format(persona_num), attr_calc_nll_log)
    print("t",T,"e",e,"r",r,"w",w)
    np.save(spath+"persona={}/parameter".format(persona_num),np.concatenate([alpha.detach(),beta.detach().numpy(),T.detach().numpy(),e.detach().numpy()],axis=0))
    np.save(spath+"persona={}/rw_paramerter".format(persona_num),np.concatenate([r.detach().numpy().reshape(1,-1),w.detach().numpy().reshape(1,-1)],axis=0))



if __name__ == "__main__":
    for skiptime in range(1,5):
        for i in range(32):
            execute_data(skiptime,i)