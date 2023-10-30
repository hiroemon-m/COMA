# Standard Library
import gc
import os
from typing import List

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from scipy import sparse
from torchviz import make_dot



# First Party Library
import csv
import config
from env import Env
from init_real_data import init_real_data


from memory import Memory
from actor import Actor
from critic import Critic
import numpy as np


import torch
import config
device = config.select_device

class COMA:
    def __init__(self, agent_num,input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = 81580
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.memory = Memory(agent_num, action_dim)
        #actorは方策なので確率が返ってくる
        #self.actors = [Actor(T,e,r,w) for _ in range(agent_num)]
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

    #観察(今いる場所)をもとに次の行動とその確立を求める
    def get_actions(self, edges,feat):
        #観察
        #print("observations",observations)


        #actions = []


            #観察の数がエージェントの数分ある
            #観察を元にactorエージェントを作成し確率を得る
            #actorはペルソナの数作る
        prob,feat,_= self.actor.predict(feat,edges)

            #一つ前の勾配の削除

 
            #print(self.actors[i])
            #print(dist)
            #エージェントiの方策の配列に追加
        #配列に追加　ノードを順に追加
  
        for i in range(self.agent_num): 
            self.memory.pi[i].append(prob[i].tolist())
            #モデルから行動を獲得
            #将来的に方策関数を拡張さえるときはこれ
            #action = Categorical(dist).sample()
            #方策関数一つなので確率一つのためベルヌーイ
        action = prob.bernoulli()
            #tensorの中身を取り出しactionsに追加
            #actions.append(action.item())
        #actions.append(action.tolist())
      
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

        #------------疎行列を作る--------------
        #edges_csr = np.empty([4,0])
        #features_csr = np.empty([4,0])
        #edges_csr = []
        #features_csr = []

         #for i in range(4):

            #edge_csr_li = np.empty([0,3])
            #feature_csr_li = np.empty([0,3])

            #edge_csr = sparse.csr_matrix(observation_edges[i])
            #feature_csr = sparse.csr_matrix(observation_features[i])

            #edge_csr_li = np.append(edge_csr_li,np.array([edge_csr.indptr]),axis=1)
            #edge_csr_li = np.append(edge_csr_li,np.array([edge_csr.indices]),axis=1)
            #edge_csr_li = np.append(edge_csr_li,np.array([edge_csr.data]),axis=1)
            #edges_csr.append([edge_csr.indptr,edge_csr.indices,edge_csr.data])
            #np.append(edge_csr,edge_csr.indptr,edge_csr.indices,edge_csr.data)
            #np.append(features_csr,feature_csr.indptr,feature_csr.indices,feature_csr.data)

            #feature_csr_li = np.append(feature_csr_li,np.array([feature_csr.indptr]),axis=1)
            #feature_csr_li = np.append(feature_csr_li,np.array([feature_csr.indices]),axis=1)
            #feature_csr_li = np.append(feature_csr_li,np.array([feature_csr.data]),axis=1)
            #features_csr.append([feature_csr.indptr,feature_csr.indices,feature_csr.data])
            
        #------------疎行列を作る--------------
   

        observation_edges_flatten = torch.tensor(np.array(observation_edges)).view(4,-1)
        observation_features_flatten = torch.tensor(np.array(observation_features)).view(4,-1)
        actions_flatten = torch.tensor(actions).view(4,-1)#4x1024
     

        for i in range(self.agent_num):
            # train actor
            #agentの数input_critic作る

            input_critic = self.build_input_critic(i, observation_edges_flatten,observation_edges, observation_features_flatten, observation_features,actions_flatten,actions)
     
            #.detach() は同一デバイス上に新しいテンソルを作成する。計算グラフからは切り離され、requires_grad=Falseになる。
            #critic_targetはQ値を行動の分返す. Q(s,(u−a, u0a))
            #Q_target = self.critic_target(input_critic).detach()
            Q_target = self.critic_target(input_critic)  

    

            #actionを列ベクトルに変形
            #print(torch.tensor(actions).shape)
         
            action_taken = torch.tensor(actions)[:,i,:].type(torch.long).reshape(4, -1)
            #print(action_taken.shape)

            #エージェントiの方策の確立とQ_targetの積
            #Q_targetに重みはかけて反証的ベースライン
            #print(pi.shape)
            #print(pi[0][i][:])
        
            #print(pi[:][i][:])
            #4x32なのでそれぞれの行で適切な値を取ってかける
            #print(Q_target)
            #dim=1で行方向に合計
            #ここ実際にagentがした行動入れていいの？
            
            baseline = torch.sum(torch.tensor(pi[i][:]) * Q_target, dim=1)
            #print(baseline)
            #print(pi[i])
            #Q値(エージェント数,行動数)から実際にエージェントのした行動を取り出す
            #squeezeで入力テンソル形状は 1  が削除されて返される,(500,1)→(500)

            # 複数の最大値がある場合

            #max_indices = torch.where(action_taken == torch.max(action_taken))
            max_indices = []
            Q_taken_target = []
            
        
         
            for j in range(4):
                Q_taken_target_num = 0
                #print(action_taken[i])
                
                max_indices.append(torch.where(action_taken[j] == 1)[0].tolist())
                actions_taken = max_indices
                #print("action",actions_taken)
                #print(Q_target[j])
                #print(len(actions_taken[j]))
                #gatherは適していない
                #Q_taken_target += torch.gather(Q_target[i], dim=0, index=torch.tensor(actions_taken))
                for k in range(len(actions_taken[j])):
                    #print(actions_taken[j][k])
                    Q_taken_target_num += Q_target[j][actions_taken[j][k]]
                Q_taken_target.append(Q_taken_target_num)
            Q_taken_target = torch.tensor(Q_taken_target)
            #Q_taken_target = torch.mul(Q_target,action_taken)
            #print(action_taken)
            #print(Q_target)
            #Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            #print("QQQQQQ")
            #print(Q_taken_target)
            #print(action_taken)
            #print(torch.gather(Q_target, dim=1, index=action_taken).size())
            #print(Q_taken_target.size())
            #print( torch.gather(Q
            # _target, dim=1, index=action_taken))
            #print(Q_taken_target)
            #利得関数
            advantage = Q_taken_target - baseline
            advantage = advantage.reshape(4,1)

            #勾配の期待値の計算
            #方策から実際に行なったな行動を取り出す。　(500,1)→(500)にしlogをとる

            #######kokokaraasita
            log_pi = torch.log(torch.mul(torch.tensor(pi[i]),action_taken))
            log_pi = torch.where(log_pi == float("-inf"),0,log_pi)

            #期待値の算出
            #print(action_taken)
            #print(torch.tensor(pi[i][0]))
            #print(log_pi.shape)
            #print(log_pi*advantage)
            actor_loss = - torch.mean(advantage * log_pi)
            #print(actor_loss)

            #image = make_dot(actor_loss, params=dict(self.actor.named_parameters()))
            #image.format = "png"
            #image.render("NeuralNet")
            #log_piに関して微分retain_graph= Tru
            actor_loss.backward(retain_graph = True)
            #このコードは、self.actors[i]（i番目のアクターネットワーク）のパラメータの勾配を計算し、その勾配のノルムが5を超える場合、ノルムを5にクリップします。これは、勾配爆発を防ぎ、訓練の安定性を向上させるために役立ちます。
            #深層強化学習では、しばしば勾配クリッピングが安定した訓練を実現するための一般的な手法として使用されます。
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
            #パラメータの更新
            actor_optimizer[i].step()

            # train critic
            #critcの行動価値関数を算出
            Q = self.critic(input_critic)
            Q_taken = []
            for j in range(4):
                Q_value = 0
                for k in range(len(actions_taken[j])):
                
                    Q_value += Q[j][actions_taken[j][k]]
                Q_taken.append(Q_value)
            Q_taken=torch.tensor(Q_taken)


            #actionを列ベクトルに変形
            #action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            #Qから実際に行動したQを取りだし、形状を(500,)に
            #Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # TD(0)


            #reward(500,2)
            #500個の0の入れるを作り新たな値を格納

            r = torch.zeros(len(reward[:, i]))
            #print(r.shape)
            #print(torch.sum(reward,dim=2).shape)
            #print("reward",reward)

            #TDλ??ではないんのでは、明日調べる
            #print(Q_taken_target)
            for t in range(len(reward[:, i])):
                #ゴールに到達した場合は,次の状態価値関数は0
        
                if t == 3:
                    r[t] = reward[:, i][t]
                #ゴールでない場合は、
                else:
                    #Reward + γV(st+1)
                    #print(t, Q_taken_target[t + 1])
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]


            #td(0)の平均二乗誤差
        
            #critic_loss = torch.mean((r - Q_taken)**2)
            r = torch.autograd.Variable(r, requires_grad=True)
            Q = torch.autograd.Variable(Q, requires_grad=True)
            critic_loss = torch.mean((r - Q_taken)**2)
            critic_optimizer.zero_grad()
            #print(critic_loss)
            #Q_taken.backward()
            #print("done_q",Q)
            ##Qが計算グラフから切り離されている
            #critic_optimizer.zero_grad()
            #r.backward(retain_graph=True)
            #critic_optimizer.zero_grad()
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
     
        #print(batch_size)
        #print(observations)
        #agent_idを要素とした配列を作る、ミニバッチ内のすべてのサンプルに対して同じ値を持つ列ベクトルを作成します。
        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)
    
     
        #edege = torch.tensor(np.array(edges))
        #feature = torch.tensor(np.array(features))
        #print(edge_csr)
        #edge_csr = torch.tensor(np.array([edge_csr[0],edge_csr[1],edge_csr[2],edge_csr[3]]))
        #feature_csr = torch.tensor(np.array([feature_csr[0],feature_csr[1],feature_csr[2],feature_csr[3]]))
        #observationsリスト内のそれぞれのエージェントの同じ時刻での観測を連結して、2次元のテンソルに変換する。obsevationsの形状を(batch_size,state_dim * .agent_num)に変換
        edges_i = torch.tensor([edges[0][agent_id],edges[1][agent_id],edges[2][agent_id],edges[3][agent_id]])
        features_i = torch.tensor([features[0][agent_id],features[1][agent_id],features[2][agent_id],features[3][agent_id]])
        
        #for i in range(4):
            #actions[i][0:agent_id] actions[i][agent_id:]
        #z    edges[i]
        
        #print("~id")
        #print(torch.tensor(actions[0][:agent_id]).shape)

        #print(torch.tensor(actions[0][agent_id+1:]).shape)
        #31*32 agent_num*agent_num
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
        
        #print(torch.cat(tensors=(torch.tensor(actions[0][:agent_id]),torch.tensor(actions[0][agent_id:])),dim=0).shape)
        #actions_i = torch.tensor([actions[0][agent_id],actions[1][agent_id],actions[2][agent_id],actions[3][agent_id]])

        #actions_iとfeatures_iが必要

        #空の分消す
   
        actions_i = actions_i[:4]
     

        #print(observation_feature.shape)
        #print(actions.shape)
        #print(actions)
        #observations = torch.concat(observation_edge,observation_feature)
        #(5x32x32,5x32x32)
        #(id, edges_flatten(t-1での行動),action_except_i[:i][i+1:](注目エージェント以外の行動,features_flatten(観測),features_faltten[i](エージェントiの観測))


        input_critic= torch.cat(tensors=(ids,edges_flatten),dim=1)
        input_critic= torch.cat(tensors=(input_critic,actions_i),dim=1)
        input_critic= torch.cat(tensors=(input_critic,features_flatten),dim=1)
        input_critic= torch.cat(tensors=(input_critic,features_i),dim=1)

        #print("observations",observations)
        # エージェントのobservationとacitonを結合する.type(torch.float32)を使用して、テンソルのデータ型を浮動小数点数に変換
        #dim=-1は一番内側の配列で結合エージェント2つの場合[観察、観察、行動、行動]
        #(5x32x32,5x32x32,5,5)
        #actionsはベルヌーイ分布で1,0返してるyお！
        #input_critic = torch.concat(observations)
        #print("input_crtic",input_critic)
        #idとinput_criticを一番内側の配列で結合
        #[エージェントid,観察、観察、行動、行動]
        #input_critic = torch.tensor([ids, observations, torch.tensor(actions)])
        #input_critic = [ids, observations, torch.tensor(actions)]
        #(5,5x32x32(エッジ),5x32x2011(属性値),5x32x32(行動))
        input_critic = input_critic.to(torch.float32)


        return input_critic
    


def execute_data():

    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)

        #alpha,betaの読み込み
    np_alpha = []
    np_beta = []
    with open("model.param.data.fast", "r") as f:
        lines = f.readlines()
        for line in lines:
            datas = line[:-1].split(",")
            np_alpha.append(np.float32(datas[0]))
            np_beta.append(np.float32(datas[1]))



    T = np.array(
        [0.8 for i in range(len(np_alpha))],
        dtype=np.float32,
    )
    e = np.array(
        [0.8 for i in range(len(np_beta))],
        dtype=np.float32,
    )
    alpha = torch.from_numpy(
        np.array(
            np_alpha,
            dtype=np.float32,
        ),
    ).to(device)

    beta = torch.from_numpy(
        np.array(
            np_beta,
            dtype=np.float32,
        ),
    ).to(device)

    r = np.array(
        [0.9 for i in range(len(np_alpha))],
        dtype=np.float32,
    )

    w = np.array(
        [1e-2 for i in range(len(np_alpha))],
        dtype=np.float32,
    )
    torch.autograd.set_detect_anomaly(True)

    N = len(np_alpha)
    del np_alpha, np_beta

    """_summary_
    setup data
    """
    LEARNED_TIME = 0
    GENERATE_TIME = 5
    TOTAL_TIME = 10

    load_data = init_real_data()

    agent_num = len(load_data.adj[LEARNED_TIME])
    
    state_dim = len(load_data.feature[LEARNED_TIME])
    input_size = 81580
    #今回は貼るor貼らないで、一つの確率で表せるので1
    action_dim = 2

    gamma = 0.99
    lr_a = 0.0001
    lr_c = 0.005

    target_update_steps = 10

    agents = COMA(agent_num, input_size, action_dim, lr_c, lr_a, gamma, target_update_steps,T,e,r,w)


    obs = Env(
        edges=load_data.adj[LEARNED_TIME].clone(),
        feature=load_data.feature[LEARNED_TIME].clone(),
        temper=T,
        #alpha=alpha,
        #beta=beta,
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