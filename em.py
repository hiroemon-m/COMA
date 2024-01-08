from numpy import linalg as LA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd





def init_gaussian_param():
    """ ガウス分布のパラメタμ, sigmaを初期化する
    Input:
    Output:
        μ: 平均値
        sigma: 共分散行列
    """
    mean = np.random.rand(1, 2) * 10
    sigma = [[1, 0], [0, 1]]
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

    with open(data, "r") as f:
        lines = f.readlines()
        #print(lines)
        for index, line in enumerate(lines):
            datas = line[:-1].split(",")
            np_alpha.append(np.float32(datas[0]))
            np_beta.append(np.float32(datas[1].replace("\n","")))

    return np_alpha,np_beta



if __name__ == "__main__": 
    path = "model.param.data.fast"
    dblp_alpha,dblp_beta = tolist(path)
    data_dblp = pd.DataFrame({"alpha":dblp_alpha,"beta":dblp_beta})
    
    transformer = MinMaxScaler()
    norm = transformer.fit_transform(data_dblp)
    data_norm = data_dblp.copy(deep=True)
    data_norm["alpha"] = norm[:,0]
    data_norm["beta"] = norm[:,1]
    alpha,beta = tolist(path)
    data = pd.DataFrame({"alpha":alpha,"beta":beta})
    dblp_array = np.array([data_norm["alpha"].tolist(),
                      data_norm["beta"].tolist()])
    dblp_array = dblp_array.T
    num = 256
    pred = KMeans(n_clusters=num).fit_predict(dblp_array)
    dblp_kmean = data_norm
    dblp_kmean["cluster_id"] = pred
    


    data = data.loc[:,"alpha":"beta"]
    # (サンプル数, 特徴量の次元数) の2次元配列で表されるデータセットを作成する。
    # 変換器を作成する。
    transformer = MinMaxScaler()
    # 変換する。
    data.loc[:,"alpha":"beta"] = transformer.fit_transform(data)

    em = 0
    li = []
    for i in data["alpha"].tolist():
        li.append([i])
    for k,j in enumerate(data["beta"].tolist()):
        li[k].append(j)


    em = np.array(li)
    path = "model.param.data.fast"
    alpha,beta = tolist(path)
    data = pd.DataFrame({"alpha":alpha,"beta":beta})
    dblp_kmean = data_norm
    dblp_kmean["cluster_id"] = pred
    print(dblp_kmean["cluster_id"].value_counts())

    mean = [[]for i in range(num)]
    sigma = []
    for i in range(num):

        mean[i].append(dblp_kmean["alpha"][dblp_kmean["cluster_id"]==i].mean())
        mean[i].append(dblp_kmean["beta"][dblp_kmean["cluster_id"]==i].mean())
        sigma.append(np.cov(dblp_kmean[dblp_kmean["cluster_id"]==i]["alpha"],dblp_kmean[dblp_kmean["cluster_id"]==i]["beta"],bias=True))

    means = []
    for i in mean:
        means.append(np.array(i))
    gamma,means = em_algorithm(500,num,em,means,sigma)
    print(gamma)
    print(np.argmax(gamma,axis=1))
    np.argmax(gamma,axis=1)
    np.save(
    "gamma{}".format(num), # データを保存するファイル名
    gamma,  # 配列型オブジェクト（listやnp.array)
    )
    np.save(
    "means{}".format(num), # データを保存するファイル名
    means,  # 配列型オブジェクト（listやnp.array)
    )
    