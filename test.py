# Standard Library
import gc
import pickle

# Third Party Library
from memory_profiler import profile
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

# First Party Library
import csv
import config
from env import Env
from init_real_data import init_real_data

#another file
from memory import Memory
from actor import Actor
from reward_policy import Model,Optimizer

device = config.select_device

class PPO:
    def __init__(self, obs,agent_num,input_size, action_dim, lr,discount ,T,e,r,w,f,rik,temperature,story_count,data_set):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = input_size
        self.discount = discount
        self.persona = rik
        self.story_count = story_count 
        self.memory = Memory(agent_num, action_dim,self.story_count,len(self.persona[0][0]),data_set)
        self.obs = obs
        self.alpha = self.obs.alpha
        self.beta = self.obs.beta
        self.gamma = self.obs.gamma
        self.ln = 0
        self.count = 0
        self.actor = Actor(T,e,r,w,f,rik,self.agent_num,temperature)
        self.new_actor = Actor(T,e,r,w,f,rik,self.agent_num,temperature)
        #adamにモデルを登録
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr) 
        self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr) 

    #@profile
    def action(self, edges,feat,time,old_feature):
        edge_prob,edge_action,feat_prob,feat_action,old_feature = self.actor.predict(feat,edges,time,old_feature)
        return edge_prob,edge_action,feat_prob,feat_action,old_feature
    
    

    #@profile
    def train(self,edge,feat,gamma,past_feature):

        cliprange=0.2
        storycount = self.story_count

        #収益の計算
        G_r = torch.empty([storycount,len(self.persona[0][0]),self.agent_num, self.agent_num])
        
        for r in range(len(self.memory.reward)-1, -1, -1):

            if r == len(self.memory.reward) - 1:
                G_r[r] = self.memory.reward[r].clone()

            else:
                G_r[r] = gamma*G_r[r+1].clone() + self.memory.reward[r].clone()

        #baselineの作成
        baseline = torch.empty([storycount,len(self.persona[0][0]),self.agent_num, self.agent_num])
        for i in range(len(self.memory.reward)):
            
            if i == 0:
                baseline[i] = self.memory.reward[i].clone()
            
            else:
                baseline[i] = (((baseline[i-1]*i)+self.memory.reward[i])/(i+1)).clone()
  
        for i in range(5):
            G,loss = 0,0
            past_feature_t = past_feature
            for i in range(storycount):
        
                old_policy = self.memory.probs[i]   
                new_policy,past_feature_t =  self.new_actor.forward(self.memory.next_features[i-1],self.memory.next_edges[i-1],i,past_feature_t)
                ratio =torch.exp(torch.log(new_policy+1e-5) - torch.log(old_policy+1e-5))
                ratio_clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
          
                #G = G_r[i] - baseline[i]
                G = G_r[i]
                reward_unclipped = ratio * G
                reward_clipped = ratio_clipped * G
                reward = torch.min(reward_unclipped, reward_clipped)
                loss = loss - torch.sum(torch.log(new_policy)*reward)
    
    

            #loss.retain_grad()
            #reward.retain_grad()
            #nw_policy.retain_grad()

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

            #for name, param in self.new_actor.named_parameters():
            #    if param.grad is not None:
            #        print(f"{i}:{name} grad: {param.grad}")
            #    else:
            #        print(f"{i}:{name} grad is None")

            self.new_actor_optimizer.step()
            
        T,e,r,w ,f= self.new_actor.T.clone().detach(),self.new_actor.e.clone().detach(),self.new_actor.r.clone().detach(),self.new_actor.W.clone().detach(),self.new_actor.f.clone().detach()
        del old_policy,new_policy,ratio,ratio_clipped,G,reward_unclipped,reward_clipped
        del reward, G_r, baseline, loss,self.new_actor, self.new_actor_optimizer, self.actor
        gc.collect()
        
        return T,e,r,w,f
    
   

def init_mixing_param(K):
    """ 混合率の初期化
    Input:
        K: the number of mixing
    Output:
        pi: the ratio of each mixing
    """
    pi = np.random.dirichlet([1] * K)
    return pi


def calc_gaussian_prob(x, mean, sigma):
    """ サンプルxが多次元ガウス分布から生成される確率を計算 """
    x = np.matrix(x)
    mean = np.matrix(mean)


    sigma = np.matrix(sigma) +  np.eye(sigma.shape[0]) * 1e-5  # 数値安定性のための小さなノイズ
    d = x - mean


  
    sigma_inv = np.linalg.inv(sigma)  # 逆行列の計算
    a = np.sqrt((2 * np.pi) ** sigma.ndim * np.linalg.det(sigma)) + 1e-20
    b = np.exp(-0.5 * (d * sigma_inv * d.T).item()) + 1e-20  # .item() はスカラー値を取得するために使用

    return b / a


def calc_likelihood(N,X, means, sigmas, pi,K):
    """ データXの現在のパラメタにおける対数尤度を求める
    """
    likehood = 0.0
    for n in range(N):
        temp = 0.0
        for k in range(K):
            temp += pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k])
    
        
        likehood += np.log(temp + 1e-10)
    return likehood


def e_step(N,K,x,sigmas,means,pi):

    X = np.array(x)
    likelihood = calc_likelihood(N,X, means, sigmas, pi, K)
    gamma = np.zeros((N, K))
    is_converged = True
    iteration = 0

    while is_converged:
        # E-Step
        for n in range(N):
            denominator = sum(pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k]) for k in range(K))
            for k in range(K):
                
                gamma[n, k] = pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k]) / denominator


        # 収束判定
        new_likelihood = calc_likelihood(N,X, means, sigmas, pi, K)
        if abs(new_likelihood - likelihood) < 0.01 or iteration >= 20:
            is_converged = False

        likelihood = new_likelihood
        iteration += 1

    return gamma

#@profile
def update_reward(agent,persona_ration,sc_mean,sc_var,optim_alpha,optim_beta,optim_gamma,time):

    #各エージェントの報酬関数のパラメータを最適化する
    alpha = torch.tensor(optim_alpha).to(device)

    beta = torch.tensor(optim_beta).to(device)

    gamma = torch.tensor(optim_gamma).to(device)
    #alpha = torch.from_numpy(np.array([1.0 for i in range(agent.agent_num)],dtype=np.float32,),).to(device)
    #beta = torch.from_numpy(np.array([1.0 for i in range(agent.agent_num)],dtype=np.float32,),).to(device)
    #gamma = torch.from_numpy(np.array([1.0 for i in range(agent.agent_num)],dtype=np.float32,),).to(device)

    #報酬関数のパラメータの更新
    reward_model = Model(alpha,beta,gamma)
    reward_optimizer = Optimizer(agent.memory,persona_ration,reward_model)
    reward_optimizer.optimize(time)
   

    
    #SGD
    update_alpha,update_beta,update_gamma = reward_model.alpha ,reward_model.beta,reward_model.gamma

    #GD
    #if time == 0:
    #    reward_model.gamma.grad = torch.zeros([32])
    #print("grad",reward_model.alpha.grad,reward_model.beta.grad,reward_model.gamma.grad)
    #lr = 0.005
    #update_alpha,update_beta,update_gamma = alpha - reward_model.alpha.grad*lr , beta - reward_model.beta.grad*lr, gamma - reward_model.gamma.grad*lr


    #更新したパラメータの標準化
    alpha_li = []
    beta_li = []
    gamma_li = []
    agent_num = agent.agent_num
    for i in range(agent_num):

        alpha_li.append(update_alpha[i].tolist())
        beta_li.append(update_beta[i].tolist())
        gamma_li.append(update_gamma[i].tolist())

    df =  pd.DataFrame({"alpha":alpha_li,"beta":beta_li,"gamma":gamma_li})

    #print("df前",df)

    #更新前データの平均、分散
    sc_mean = torch.tensor(sc_mean)
    sc_var = torch.tensor(sc_var)

    #更新後データの平均、分散
    add_mean = torch.tensor([df.mean()[0],df.mean()[1],df.mean()[2]])
    add_var = torch.tensor([df.var()[0],df.var()[1],df.var()[2]])

    #平均の更新
    new_mean = (add_mean*agent_num + sc_mean*agent_num)/(2*agent_num)
    #分散の更新
    orginal_data = sc_var+(sc_mean**2)
    add_data = add_var+(add_mean**2)
    sqrt_mean = ((orginal_data)*agent_num + (add_data)*agent_num)/(2*agent_num)
    new_var = (sqrt_mean) - (new_mean**2)
    #tensor to Numpy
    new_mean = new_mean.cpu().detach().numpy().copy()
    new_var = new_var.cpu().detach().numpy().copy()
    #標準化
    #X_new_scaled = (df - new_mean) / np.sqrt(new_var)
    #標準化
    sc = StandardScaler()
    X_new_scaled = sc.fit_transform(df)

    #print("df後", X_new_scaled)
    #元データで標準化
    #update_alpha =  X_new_scaled["alpha"]
    #update_beta =  X_new_scaled["beta"]
    #update_gamma =  X_new_scaled["gamma"]
    #sc
    update_alpha =  X_new_scaled[:,0]
    update_beta =  X_new_scaled[:,1]
    update_gamma =  X_new_scaled[:,2]
    reward_param = []
    #print("更新後",X_new_scaled)

    for i in range(agent.agent_num):
        reward_param.append([update_alpha[i]
                             ,update_beta[i]
                             ,update_gamma[i]])
    reward_param = np.array(reward_param)
    return reward_param,update_alpha,update_beta,update_gamma

def tolist(data_path) -> None:
    alpha_li = []
    beta_li = []
    gamma_li = []

    with open(data_path, "r") as f:
        lines = f.readlines()      
        for index, line in enumerate(lines):
            load_datas = line[:-1].split(",")
            alpha_li.append(np.float32(load_datas[0]))
            beta_li.append(np.float32(load_datas[1]))
            gamma_li.append(np.float32(load_datas[2]))

    return alpha_li,beta_li,gamma_li

#@profile
def execute_data(persona_num,data_name):
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

    #パラメータの読み込み
    path_n = "gamma/{}/persona={}/".format(data_name,persona_num)
    #ペルソナの割り当て率:timexagent_numxpersona_num
    path = path_n+"gamma{}.npy".format(persona_num)
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    persona_ration = torch.from_numpy(np.tile(persona_ration,(story_count,1,1))).to(device)
    #各ペルソナの平均値
    path = path_n+"means{}.npy".format(persona_num)
    means = np.load(path)
    means = means.astype("float32")
    #means = torch.from_numpy(means).to(device)
    #各ペルソナの混合率
    path = path_n+"pi{}.npy".format(persona_num)
    pi = np.load(path)
    pi = pi.astype("float32")
    #pi = torch.from_numpy(pi).to(device)
    #各ペルソナの分散
    path = path_n+"sigma{}.npy".format(int(persona_num))
    sigma = np.load(path)
    sigma = sigma.astype("float32")
    #sigma = torch.from_numpy(sigma).to(device)
    #normのためのmodelのロード
    path_norm = path_n+"norm"

    #重み(固定値)
    alpha = torch.tensor(means[:,0]).unsqueeze(1)
    beta = torch.tensor(means[:,1]).unsqueeze(1)
    gamma = torch.tensor(means[:,2]).unsqueeze(1)

    #t=4でのパラメータ
    path = "gamma/{}/optimized_param".format(data_name)
    
    print("mean",means)

    print("alpha",alpha)
    print("beta",beta)
    print("gamma",gamma)

    #パラメータ
    mu = 0.8922
    lr = 0.01
    temperature = 0.05 
    T = torch.tensor([[[1.0]] for _ in range(persona_num)], dtype=torch.float32)
    e = torch.tensor([[[1.0]] for _ in range(persona_num)], dtype=torch.float32)
    r = torch.tensor([[[0.7]] for _ in range(persona_num)], dtype=torch.float32)
    w = torch.tensor([[[1.0]] for _ in range(persona_num)], dtype=torch.float32)
    f = torch.tensor([[[1.0]] for _ in range(persona_num)], dtype=torch.float32)
  
    #初期化
    flag = True
    episode = 0
    episodes_reward = []
    old_feature = torch.empty(agent_num,input_size,requires_grad=False)
    for i in range(5):
        old_feature  = 0.5*old_feature + load_data.feature[i]

    print("sum",torch.sum(load_data.adj[LEARNED_TIME],dim=-1))
    print("sum",torch.sum(load_data.feature[LEARNED_TIME],dim=-1))


    while flag:
        print("--------------------episode:{}--------------------".format(episode))
        norm = pickle.load(open(path_norm,"rb"))
        sc_mean = norm.mean_
        sc_var = norm.var_
        sc = agent_num

        # E-step 
        #初期値はt=0~4のデータを元に設定
        if episode < 10:
            mixture_ratio = persona_ration

       #t=5~10の生成されたネットワークを元に割り当て率の変更
        else:
            optim_alpha,optim_beta,optim_gamma = tolist(path)
            for t in range(5):
                
                reward_param,optim_alpha,optim_beta,optim_gamma =update_reward(agents,persona_ration,sc_mean,sc_var,optim_alpha,optim_beta,optim_gamma,t)
                ration = e_step(agents.agent_num,persona_num,reward_param,sigma,means,pi).astype("float32")
                persona_ration[t] = torch.from_numpy(ration)

            mixture_ratio = persona_ration
                 
            del agents
            print("persona_ration",mixture_ratio)
        #personaはじめは均等
        
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
     
        
        agents = PPO(obs,agent_num, input_size, action_dim,lr, mu,T,e,r,w,f,mixture_ratio,temperature,story_count,data_name)
        episode_reward = 0
      

        past_feature = old_feature
        test_past_feature = old_feature
        agents.memory.next_edges[0],agents.memory.next_features[0]=obs.state()

        for time in range(story_count):
            #初めは,32,32
            #2回目は、ペルソナ毎に遷移が違う
            edge,feature = obs.state()
            edge_prob,edge_action,feat_prob,feat_action,past_feature= agents.action(edge,feature,time,past_feature)
         

            #属性値を確率分布の出力と考えているので、ベルヌーイ分布で値予測
            #ペルソナ毎のエッジとペルソナ毎の属性値を与えたい
            reward = obs.step(feat_action,edge_action,time)
            #memoryはtrainで使う
            agents.memory.probs[time]=edge_prob.clone()
            agents.memory.feat_probs[time]=feat_prob.clone()
            agents.memory.next_edges[time + 1]=edge_action.clone()
            agents.memory.next_features[time + 1]=feat_action.clone()
            agents.memory.reward[time]=reward.detach().clone()
            episode_reward = episode_reward + reward.clone().detach().sum()
            del edge,feature,edge_prob,edge_action,feat_action,reward
            gc.collect()
      

        episodes_reward.append(episode_reward)
        print("epsiode_rewaerd",episodes_reward[-1])

        T,e,r,w,f= agents.train(load_data.adj[LEARNED_TIME].clone(),load_data.feature[LEARNED_TIME].clone(),mu,test_past_feature)
        print("パラメータ",T,e,r,w,f)
        del past_feature,obs
        gc.collect()
        episode += 1
  
        #print("persona_ration",mixture_ratio)

        if episode % 10 == 0:
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")

        if episode >=20:
            flag = False

        else:
            del episode_reward
            gc.collect()

 


    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    print("学習終了")
    print("パラメータ",T,e,r,w,f)
    optim_alpha,optim_beta,optim_gamma = tolist(path)
    for t in range(5):
        reward_param,optim_alpha,optim_beta,optim_gamma =update_reward(agents,persona_ration,sc_mean,sc_var,optim_alpha,optim_beta,optim_gamma,time)
        ration = e_step(agents.agent_num,persona_num,reward_param,sigma,means,pi).astype("float32")
        print("rrrrrrrrra",ration)
        persona_ration[t] = torch.from_numpy(ration)

    mixture_ratio = persona_ration
   
    obs = Env(
        agent_num=agent_num,
        edges=load_data.adj[LEARNED_TIME].clone(),
        feature=load_data.feature[LEARNED_TIME].clone(),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        persona=mixture_ratio
        )
    
    agents = PPO(obs,agent_num, input_size, action_dim,lr, gamma,T,e,r,w,f,mixture_ratio,temperature,story_count,data_name)

        
    for count in range(10):

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
            print("pred feat sum",torch.sum(feat_action))
            print("pred feat sum",torch.sum(feat_action,dim=1))
            print("target feat sum",attr_test[attr_test>0].sum())
            

            try:    
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_test),
                )
                #plot
                auc_actv = roc_auc_score(attr_test, attr_predict_probs)
                fpr_a ,tpr_a, _ = roc_curve(1-attr_test, 1-attr_predict_probs)

                plt.figure()
                plt.plot(fpr_a, tpr_a, marker='o')
    
                plt.xlabel('FPR: False positive rate')
                plt.ylabel('TPR: True positive rate')

                plt.grid()
                plt.savefig('persona={}sklearn_roc_curve.png'.format(persona_num))
               
            finally:
                print("attr auc, t={}:".format(test_time), auc_actv)

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

            criterion = nn.CrossEntropyLoss()
            error_edge = criterion(
                torch.from_numpy(edge_predict_probs),
                torch.from_numpy(edge_test),
            )
            auc_calc = roc_auc_score(edge_test, edge_predict_probs)
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
    for i in [5,8,12,16]:
        execute_data(i,"NIPS")