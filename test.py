# Standard Library
import gc

# Third Party Library
from memory_profiler import profile
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

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
        self.memory = Memory(agent_num, action_dim,self.story_count,len(self.persona[0][0]),data_set)
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

    #@profile
    def get_actions(self, edges,feat,time,old_feature):
        
        """ 行動確率と属性値を計算する
        Input:エッジ,属性値
        Output:
            prob エッジ方策確率
            feat 属性値
        """
        edge_prob,edge_action,feat_prob,feat_action,old_feature = self.actor.predict(feat,edges,time,old_feature)
        return edge_prob,edge_action,feat_prob,feat_action,old_feature
    
    

    #@profile
    def train(self,edge,feat,gamma,past_feature):
        """ 訓練用の関数
        Input:割引率
        Output:
            
        """
        
        cliprange=0.2
        storycount = self.story_count
        lnpxz = self.memory.probs.view(-1).sum()
        G_r = torch.empty([storycount,len(self.persona[0][0]),self.agent_num, self.agent_num])
        #self.memory 10x32x1
        #収益の計算
        baseline = torch.empty([storycount,len(self.persona[0][0]),self.agent_num, self.agent_num])

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
  
        for i in range(5):
            G,loss = 0,0
            past_feature_t = past_feature
            for i in range(storycount):
        
                old_policy = self.memory.probs[i]   
                new_policy,past_feature_t =  self.new_actor.forward(self.memory.features[i],self.memory.edges[i],i,past_feature_t)
                #new_policy,past_feature_t =  self.new_actor.forward(edge,feat,i,past_feature_t)
                ratio =torch.exp(torch.log(new_policy+1e-1) - torch.log(old_policy+1e-1))
                ratio_clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
                #G = G_r[i] - baseline[i]
                G = G_r[i] - baseline[i]
                reward_unclipped = ratio * G
                reward_clipped = ratio_clipped * G
                reward = torch.min(reward_unclipped, reward_clipped)

                # 最大化のために-1を掛ける
                #print("Er",torch.sum(torch.log(new_policy[i])))
                #print("ra",ratio,ratio_clipped)
                #print("ra",torch.sum(torch.isnan(ratio)),torch.sum(torch.isnan(ratio_clipped)))
                #print("reward",torch.sum(reward_unclipped),torch.sum(reward_clipped))
                #デバッグ
                #print("new*reward",torch.sum(torch.isnan(torch.log(new_policy)*reward)))
                #print("new",torch.sum(torch.isnan(torch.log(new_policy))))
                #print("reward",torch.sum(torch.isnan(reward)))
                loss = loss - torch.sum(torch.log(new_policy)*reward)

            loss.retain_grad()
            reward.retain_grad()
            new_policy.retain_grad()


            self.new_actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)
                        # 勾配の計算と適用
            #print("a",new_policy.grad)
            #print("r",reward.grad)
            #print("loss",loss.grad)
            #print("loss",loss)
            
            #for param in self.new_actor.parameters():
            #    if param.grad is not None:
            #        param.grad.data = param.grad.data / (param.grad.data.norm() + 1e-6)

            #torch.nn.utils.clip_grad_norm_(self.new_actor.parameters(), 1000)

            for name, param in self.new_actor.named_parameters():
                if param.grad is not None:
                    print(f"{i}:{name} grad: {param.grad}")
                else:
                    print(f"{i}:{name} grad is None")

            self.new_actor_optimizer.step()
            print("update",self.new_actor.T,self.new_actor.e,self.new_actor.r,self.new_actor.W)
            
        T,e,r,w = self.new_actor.T.clone().detach(),self.new_actor.e.clone().detach(),self.new_actor.r.clone().detach(),self.new_actor.W.clone().detach()
        del old_policy,new_policy,ratio,ratio_clipped,G,reward_unclipped,reward_clipped
        del reward, G_r, baseline, loss,self.new_actor, self.new_actor_optimizer, self.actor
        gc.collect()
        
        return T,e,r,e
    
#@profile
def e_step(agent_num,load_data,T,e,r,w,persona,step,base_time,temperature,old_feature):

    actor = Actor(T,e,r,w,persona,agent_num,temperature)
    past_feature = old_feature

    policy_prob = actor.calc_ration(
                load_data.feature[base_time].clone(),
                load_data.adj[base_time].clone(),
                persona,
                past_feature
                )
    


    #時間に対して縮約 5,4,32,32 -> 5,4,32,32
    top = policy_prob

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
                original = ration[m,l,n]
                rik[m,n,l] = original.detach()
    
    del actor,policy_prob,top,bottom,ration,original
    gc.collect()
    return rik
    

#@profile
def show():
    print("dell")

#@profile
def execute_data(persona_num,data_name):
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
    #alpha,betaの読み込み
    if data_name == "NIPS":
        action_dim = 32
    else:
        action_dim = 500

    LEARNED_POLICY_TIME = 4
    LEARNED_TIME = 0
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    story_count = 5
    load_data = init_real_data()
    agent_num = len(load_data.adj[LEARNED_POLICY_TIME])
    input_size = len(load_data.feature[LEARNED_POLICY_TIME][1])

    path_n = "gamma/{}/".format(data_name)
    persona_ration = []
    for t in range(story_count):
        path = path_n+"gamma{}_{}.npy".format(int(persona_num),t)
        persona_ration.append(np.load(path).astype("float32"))
 
    #5,32,4
    persona_ration = torch.tensor(persona_ration).to(device)
    print("size",persona_ration.size())
    #persona_ration = torch.from_numpy(np.tile(persona_ration,(story_count,1,1))).to(device)
    means = []
    for t in range(story_count):
        path = path_n+"means{}_{}.npy".format(int(persona_num),t)
        means_value = np.load(path).astype("float32")
        means.append(means_value)

    means = torch.tensor(means).to(device)
    print("means",means.size())
    #重み(固定値)
    
    alpha = torch.unsqueeze(torch.tensor(means[:,:,0]),dim=-1)
    beta = torch.unsqueeze(torch.tensor(means[:,:,1]),dim=-1)
    gamma = torch.unsqueeze(torch.tensor(means[:,:,2]),dim=-1)

    print("alpha",alpha)
    print("beta",beta)
    print("gamma",gamma)

    #パラメータ
    mu = 0.8922
    lr = 0.01
    temperature = 0.05


    
    T = torch.tensor([[[1.0]] for _ in range(persona_num)], dtype=torch.float32)
    e = torch.tensor([[[1.0]] for _ in range(persona_num)], dtype=torch.float32)
    r = torch.tensor([[[0.75]] for _ in range(persona_num)], dtype=torch.float32)
    w = torch.tensor([[[1.0]] for _ in range(persona_num)], dtype=torch.float32)
  
    print("T.size",T.size())
    ln = 0
    ln_sub = 0

    flag = True
    episode = 0
    episodes_reward = []

    old_feature = torch.empty(agent_num,input_size,requires_grad=False)

    for i in range(5):
        old_feature  = 0.8*old_feature + load_data.feature[i]


    while flag and ln_sub <= 1:
        
        print("----------episode:{}----------".format(episode))


        #mixture_ratio:混合比率    
        mixture_ratio = persona_ration

        #環境の設定
        obs = Env(
            agent_num=agent_num,
            edges=load_data.adj[LEARNED_POLICY_TIME].clone(),
            feature=load_data.feature[LEARNED_POLICY_TIME].clone(),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            persona=mixture_ratio
        )

        agents = PPO(obs,agent_num, input_size, action_dim,lr, mu,T,e,r,w,mixture_ratio,temperature,story_count,data_name)
        episode_reward = 0
        past_feature = old_feature
        test_past_feature = old_feature

        for time in range(story_count):
            edge,feature = obs.state()
            #いらんかもedge_action,feat_action
            #persona_edge persona_feat　用意してそれぞれの
            edge_prob,edge_action,feat_prob,feat_action,past_feature= agents.get_actions(edge,feature,time,past_feature)
         
            reward = obs.step(feat_action,edge_action,time)
            agents.memory.probs[time]=edge_prob.clone()
            agents.memory.edges[time] = edge.clone()
            agents.memory.features[time] = feature.clone()
            agents.memory.next_edges[time]=edge_action.clone()
            agents.memory.next_features[time]=feat_action.clone()
            agents.memory.reward[time]=reward.detach().clone()
            episode_reward = episode_reward + reward.clone().detach().sum()
            del edge,feature,edge_prob,edge_action,feat_action,reward
            gc.collect()
    

        episodes_reward.append(episode_reward)
        print("epsiode_rewaerd",episodes_reward[-1])

        T,e,r,w= agents.train(load_data.adj[LEARNED_TIME].clone(),load_data.feature[LEARNED_TIME].clone(),mu,test_past_feature)
        print("パラメータ",T,e,r,w)
        del past_feature
        gc.collect()
  


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

        if episode >=20:
            flag = False

        else:
            print("delete")
            show()
            del episode_reward
            del agents.memory,agents,obs
            gc.collect()
            show()

        obs = Env(
            agent_num=agent_num,
            edges=load_data.adj[LEARNED_TIME].clone(),
            feature=load_data.feature[LEARNED_TIME].clone(),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            persona=mixture_ratio
        )
    agents = PPO(obs,agent_num, input_size, action_dim,lr, gamma,T,e,r,w,mixture_ratio,temperature,story_count,data_name)



    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    print("学習終了")
    print("パラメータ",T,e,r,w)

            
    #new_mixture_ratio = e_step(
    #            agent_num=agent_num,
    #            load_data=load_data,
    #            T=T,
    #            e=e,
    #            r=r,
    #            w=w,
    #            persona=persona_ration,
    #            step = story_count,
    #            base_time=LEARNED_TIME,
    #            temperature=temperature,
    #            old_feature=old_feature
    #        )

                      

    # スムージングファクター
    #
    # a = 0.1
    #print("nm",new_mixture_ratio)
    #updated_prob_tensor = (1 - a) * mixture_ratio + a * new_mixture_ratio
    
    agents = PPO(obs,agent_num, input_size, action_dim,lr, gamma,T,e,r,w,mixture_ratio,temperature,story_count,data_name)


        
    for count in range(3):

        obs.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
            persona=mixture_ratio
        )

        past_feature = old_feature
   
        for test_time in range(TOTAL_TIME - GENERATE_TIME):

            edges, feature = obs.state()
            edge_action,edge_prob ,feat_prob ,feat_action,past_feature = agents.actor.test(edges,feature,test_time,past_feature)
            reward = obs.test_step(feat_action,edge_action,test_time)

            #属性値の評価 
            pred_prob = torch.ravel(feat_prob).to("cpu")
            pred_prob = pred_prob.to("cpu").detach().numpy()

            detach_attr = (
                torch.ravel(load_data.feature[GENERATE_TIME + test_time])
                .detach()
                .to("cpu")
            )

            detach_attr[detach_attr > 0] = 1.0
            pos_attr = detach_attr.numpy()
            attr_test = np.concatenate([pos_attr], 0)
            attr_predict_probs = np.concatenate([pred_prob], 0)
            print("属性値の総数")

            #print("pred feat sum",attr_predict_probs.reshape((500, -1)).sum(axis=1))
            #print("target edge sum",attr_test.reshape((500, -1)).sum(axis=1))s
            print("pred feat sum",torch.sum(feat_action))
            print("pred feat sum",torch.sum(feat_action,dim=1))
            print("target feat sum",attr_test[attr_test>0].sum())
            print("target feat sum",torch.sum(load_data.feature[GENERATE_TIME + test_time]>0,dim=1))
            


            try:
                
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_test),
                )
               
                precision, recall, _ = precision_recall_curve(attr_test, attr_predict_probs)
                plt.figure()
                plt.plot(precision, recall, marker='o')
    
                plt.xlabel('FPR: False positive rate')
                plt.ylabel('TPR: True positive rate')

                plt.grid()
                plt.savefig('persona={}sklearn_pr_curve.png'.format(persona_num))
                pr_auc = auc(recall, precision)
                auc_actv = roc_auc_score(attr_test, attr_predict_probs)
 
                fpr_a ,tpr_a, thresholds = roc_curve(1-attr_test, 1-attr_predict_probs)

                plt.figure()
                plt.plot(fpr_a, tpr_a, marker='o')
    
                plt.xlabel('FPR: False positive rate')
                plt.ylabel('TPR: True positive rate')

                plt.grid()
                plt.savefig('persona={}sklearn_roc_curve.png'.format(persona_num))
               
            finally:
                print("attr auc, t={}:".format(test_time), auc_actv)
                #print("attr auc, t={}:".format(test_time), pr_auc)
                #print("attr nll, t={}:".format(t), error_attr.item())
                #attr_calc_log[count][test_time] = auc_actv
                attr_calc_log[count][test_time] = auc_actv
                attr_calc_nll_log[count][test_time] = error_attr.item()
            
      

            #エッジの評価

            #予測データ
            target_prob= edge_prob
            #print("pi",pi_test)     
            target_prob = target_prob.view(-1)
            target_prob = target_prob.to("cpu").detach().numpy()
            edge_predict_probs = np.concatenate([target_prob], 0)
            
            #テストデータ
            detach_edge = (
                torch.ravel(load_data.adj[GENERATE_TIME + test_time])
                .detach()
                .to("cpu")
            )

            pos_edge = detach_edge.numpy()
            edge_test = np.concatenate([pos_edge], 0)

            #print("エッジの総数")
            #print("pred feat sum",edge_predict_probs.reshape((500, -1)).sum(axis=1))
            #print("target edge sum",edge_test.reshape((500, -1)).sum(axis=1))
            #print("pred edge sum",edge_predict_probs.sum())
            #print("acction edge sum",edge_action.sum())
            #print("target edge sum",edge_test.sum())

            criterion = nn.CrossEntropyLoss()
            error_edge = criterion(
                torch.from_numpy(edge_predict_probs),
                torch.from_numpy(edge_test),
            )
            auc_calc = roc_auc_score(edge_test, edge_predict_probs)
            #precision, recall, _ = precision_recall_curve(edge_test, edge_predict_probs)
            #auc_calc = auc(recall, precision)
            fpr, tpr, thresholds = roc_curve(edge_test, edge_predict_probs)
            plt.clf()
            plt.plot(fpr, tpr, marker='o')
           
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.grid()
            plt.savefig('perosna={}edge_sklearn_roc_curve.png'.format(persona_num))
            #print("-------")
            print("edge auc, t={}:".format(test_time), auc_calc)
            #print("edge nll, t={}:".format(t), error_edge.item())
            #print("-------")
            calc_log[count][test_time] = auc_calc
            calc_nll_log[count][test_time] = error_edge.item()
            agents.memory.clear()

    

    np.save("experiment_data/{}/param/persona={}/proposed_edge_auc".format(data_name,persona_num), calc_log)
    np.save("experiment_data/{}/param/persona={}/proposed_edge_nll".format(data_name,persona_num), calc_nll_log)
    np.save("experiment_data/{}/param/persona={}/proposed_attr_auc".format(data_name,persona_num), attr_calc_log)
    np.save("experiment_data/{}/param/persona={}/proposed_attr_nll".format(data_name,persona_num), attr_calc_nll_log)
    print("t",T,"e",e,"r",r,"w",w)
    print(mixture_ratio)
    np.save("experiment_data/{}/param/persona={}/persona_ration".format(data_name,persona_num),np.concatenate([mixture_ratio.detach().numpy()],axis=0))
    np.save("experiment_data/{}/param/persona={}/paramerter".format(data_name,persona_num),np.concatenate([T.detach().numpy(),e.detach().numpy(),r.detach().numpy(),w.detach().numpy()],axis=0))


  




if __name__ == "__main__":
    #[5,8,12,16,24,32,64,128]
    #[4,8,12,16]
    for i in [5]:
        execute_data(i,"NIPS")