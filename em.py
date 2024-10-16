from numpy import linalg as LA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
import pandas as pd





def init_gaussian_param():
    """ ガウス分布のパラメタμ, sigmaを初期化する
    Input:
    Output:
        μ: 平均値
        sigma: 共分散行列
    """
    mean = np.random.rand(1, 3) * 10
    sigma = [[1,0,0], [0,1,0],[0,0,1]]
    return mean, sigma


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
    sigma = np.matrix(sigma) + np.eye(sigma.shape[0]) * 1e-5  # 数値安定性のための小さなノイズ
    d = x - mean
    sigma_inv = np.linalg.inv(sigma)  # 逆行列の計算
    a = np.sqrt((2 * np.pi) ** sigma.ndim * np.linalg.det(sigma))
    b = np.exp(-0.5 * (d * sigma_inv * d.T).item())  # .item() はスカラー値を取得するために使用
    return b / a


def calc_likelihood(X, means, sigmas, pi,K):
    """ データXの現在のパラメタにおける対数尤度を求める
    """
    likehood = 0.0
    N = len(X)
    for n in range(N):
        temp = 0.0
        for k in range(K):
            temp += pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k])
        likehood += np.log(temp)
    return likehood


def em_algorithm(N, K, X, means, sigmas):
    pi = init_mixing_param(K)
    likelihood = calc_likelihood(X, means, sigmas, pi, K)
    gamma = np.zeros((N, K))
    is_converged = False
    iteration = 0

    while not is_converged:
        # E-Step
        for n in range(N):
            denominator = sum(pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k]) for k in range(K))
            for k in range(K):
                gamma[n, k] = pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k]) / denominator

        # M-Step
        Nks = gamma.sum(axis=0)
        for k in range(K):
            means[k] = np.sum(gamma[:, k, np.newaxis] * X, axis=0) / Nks[k]
            diff = X - means[k]
            sigmas[k] = np.dot(gamma[:, k] * diff.T, diff) / Nks[k]
            pi[k] = Nks[k] / N

        # 収束判定
        new_likelihood = calc_likelihood(X, means, sigmas, pi, K)
        if abs(new_likelihood - likelihood) < 0.01 or iteration >= 20:
            is_converged = True
        print(likelihood)
        print(new_likelihood)
        likelihood = new_likelihood
        iteration += 1


    return gamma, means



def tolist(data) -> None:
    np_alpha = []
    np_beta = []
    np_gamma = []

    with open(data, "r") as f:
        lines = f.readlines()
        
        #print(lines)
        for index, line in enumerate(lines):
            datas = line[:-1].split(",")
            print(datas[2])
            np_alpha.append(np.float32(datas[0]))
            np_beta.append(np.float32(datas[1]))
            np_gamma.append(np.float32(datas[2]))

    return np_alpha,np_beta,np_gamma



if __name__ == "__main__":
    data_name = "Reddit" 
    if data_name == "DBLP":
        persona_list = [5,25,50]
    if data_name == "NIPS":
        persona_list = [3,5,8,12,16]
    if data_name == "Twitter":
        persona_list = [5,20,50,100]
    if data_name == "Reddit":
        persona_list = [5,20,50,100,200]
    for k in persona_list:
        #データの読み込み
        path = "optimize/complete/{}/model_param".format(data_name)
        dblp_alpha,dblp_beta,dblp_gamma = tolist(path)
        #df化
        data_dblp = pd.DataFrame({"alpha":dblp_alpha,"beta":dblp_beta,"gamma":dblp_gamma})
        #標準化
        transformer = StandardScaler()
        norm = transformer.fit_transform(data_dblp)
        data_norm = pd.DataFrame(norm, columns=["alpha", "beta", "gamma"])

        #k-means
        num = k
        pred = KMeans(n_clusters=num).fit_predict(data_norm.to_numpy())
        dblp_kmean = data_norm.copy()
        dblp_kmean["cluster_id"] = pred


        N = len(dblp_alpha)
        em = data_norm.to_numpy()
        mean = [[]for i in range(num)]
        sigma = []
        for i in range(num):

            mean[i].append(dblp_kmean["alpha"][dblp_kmean["cluster_id"]==i].mean())
            mean[i].append(dblp_kmean["beta"][dblp_kmean["cluster_id"]==i].mean())
            mean[i].append(dblp_kmean["gamma"][dblp_kmean["cluster_id"]==i].mean())

            cluster_data = dblp_kmean[dblp_kmean["cluster_id"] == i][["alpha", "beta", "gamma"]]
            if not cluster_data.empty:
                # np.covは2次元データを期待するので、3次元データの場合は転置して与える
                cov_matrix = np.cov(cluster_data.T, bias=True)
                sigma.append(cov_matrix)

        means = []
        for i in mean:
            means.append(np.array(i))



        gamma,means = em_algorithm(N,num,em,means,sigma)
        #original_means = transformer.inverse_transform(means)
        print(dblp_kmean["cluster_id"])
        print(mean)
        print(sigma)
        print(gamma)
        print(np.argmax(gamma,axis=1))

        np.argmax(gamma,axis=1)
        np.save(
            
        "optimize/complete/{}/persona={}/gamma".format(data_name,num), # データを保存するファイル名
        gamma,  # 配列型オブジェクト（listやnp.array)
        )
        np.save(
        "optimize/complete/{}/persona={}/means".format(data_name,num), # データを保存するファイル名
        means,  # 配列型オブジェクト（listやnp.array)
        )
        