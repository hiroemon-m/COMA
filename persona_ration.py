import numpy as np
import torch
# First Party Library
import config
import torch.optim as optim
import torch.nn as nn
device = config.select_device


class Model(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()

        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)
        return


class Optimizer:
    def __init__(self, edges, feats, model: Model, size: int):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)

        return

    def optimize(self, t: int):
     
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device) 
        self.optimizer.zero_grad()
        norm = feat.norm(dim=1)[:, None] + 1e-8
        feat_norm = feat.div(norm)
        dot_product = torch.matmul(feat_norm, torch.t(feat_norm)).to(device)
        sim = torch.mul(edge, dot_product)
        sim = torch.mul(sim, self.model.alpha)
        sim = torch.add(sim, 0.001)
        costs = torch.mul(edge, self.model.beta)
        costs = torch.add(costs, 0.001)
        reward = torch.sub(sim, costs)   

        if t > 0:
        
            new_feature = torch.matmul(self.edges[t],self.feats[t])
            old_feature = torch.matmul(self.edges[t-1], self.feats[t-1])
            reward += torch.sum((
                torch.matmul(self.model.gamma.view(1,-1),(torch.abs(new_feature - old_feature)+1e-4)))        
            )

        loss = -reward.sum()
        loss.backward()
        self.optimizer.step()

    def export_param(self,data):
        with open("gamma/{}/optimized_param".format(data), "w") as f:
            max_alpha = 1.0
            max_beta = 1.0
            max_gamma = 1.0

            for i in range(self.size):
                f.write(
                    "{},{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                        self.model.gamma[i].item() / max_gamma,
                    )
                )


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
            temp += pi[k] * calc_gaussian_prob(X[n], np.array(means[k]), np.array(sigmas[k]))
    
        
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
def update_reward(agent,persona_ration,norm,time):

    #各エージェントの報酬関数のパラメータを最適化する
    alpha = torch.from_numpy(
        np.array([1.0 for _ in range(agent.agent_num)],dtype=np.float32,),).to(device)

    beta = torch.from_numpy(
        np.array([1.0 for _ in range(agent.agent_num)],dtype=np.float32,),).to(device)

    gamma = torch.from_numpy(
        np.array([1.0 for _ in range(agent.agent_num)],dtype=np.float32,),).to(device)

    #報酬関数のパラメータの更新

    #更新したパラメータの標準化
    np_alpha = []
    np_beta = []
    np_gamma = []
    for i in range(agent.agent_num):

        np_alpha.append(update_alpha[i].to('cpu').detach().numpy().copy())
        np_beta.append(update_beta[i].to('cpu').detach().numpy().copy())
        np_gamma.append(update_gamma[i].to('cpu').detach().numpy().copy())

    df =  pd.DataFrame({"alpha":np_alpha,"beta":np_beta,"gamma":np_gamma})
    print("alpha",np_alpha)
    print("beta",np_beta)
    transformer = norm
    df = transformer.transform(df)
    print("df",df)
    update_alpha = df[:,0]
    update_beta = df[:,1]
    update_gamma = df[:,2]
    reward_param = []

    for i in range(agent.agent_num):
        reward_param.append([update_alpha[i]
                             ,update_beta[i]
                             ,update_gamma[i]])
    return reward_param
