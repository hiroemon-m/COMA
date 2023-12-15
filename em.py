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
    """ サンプルxが多次元ガウス分布から生成される確率を計算
    """
    # 観測データをnumpy化
    x = np.matrix(x)
    mean = np.matrix(mean) + 1e-5
    sigma = np.matrix(sigma) + 1e-5
    #ガウス分布の1/√2πσ^2の計算
    #np.linalg.det行列式の計算
    #.dim次元数
    a = np.sqrt(np.linalg.det(sigma) * (2*np.pi)**sigma.ndim)

    #ガウス分布のexpの中身の計算
    #print(mean)
    #print(-0.5*(x-mean)*sigma.I*(x-mean).T)
    try:
        b = np.linalg.det((-0.5*(x-mean)*sigma.I*(x-mean).T))
    except LA.LinAlgError:
        b = np.linalg.pinv(-0.5*(x-mean)*sigma.I*(x-mean).T)
        print("error")
    #print(a)
    #print(b)
    #print(np.exp(b)/a)
    return np.exp(b)/a

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


def em_algorism(N,K,X,mean,sigma):

    # 1. パラメタ初期化
    D = 2  # nb_feature
    N = N  # nb_sample
    K = K
    means = mean
    sigmas = sigma

    pi = init_mixing_param(K)
    print("## Initialization")
    print("mean: %s" % str(means))
    print("covariance: %s" % str(sigmas))
    print("pi: %s" % str(pi))
    print("\n")


    likehood = calc_likelihood(X, means, sigmas, pi,K)
    gamma = np.zeros((N, K))
    print("--------")
    print(likehood)
    is_converged = False
    iteration = 0
    while not is_converged:
        print("likehood: %f" % (likehood))
        # 2. E-Step: 現在のパラメタを使ってgammaを計算
        for n in range(N):
            denominator = 0.0
            # 分母を計算
            for j in range(K):
                denominator += pi[j] * calc_gaussian_prob(X[n], means[j], sigmas[j])
            # 各kについての負担率を計算
            for k in range(K):
                gamma[n, k] = pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k]) / denominator
        print("a")
        # 3. M-Step: 現在の負担率を使ってパラメタを計算
        #gamma(N,K)を行方向に加算 (N,K)→(1,K)NK
        #gamma[n][k]負担率 Nk
        Nks = gamma.sum(axis=0)
        #print("gamma",gamma)
        #print("NKs",Nks)
        for k in range(K):
            # meansを再計算
            means[k] = np.array([0.0, 0.0])
            for n in range(N):
                means[k] += gamma[n][k] * X[n]
            means[k] /= Nks[k]

            # sigmasを再計算
            sigmas[k] = np.array([[0.0, 0.0], [0.0, 0.0]])
            for n in range(N):
                _diff_vector = X[n] - means[k]
                sigmas[k] += gamma[n][k] * _diff_vector.reshape(2, 1) * _diff_vector.reshape(1, 2)

            sigmas[k] /= Nks[k]

            # piを再計算
            pi[k] = Nks[k] / N

        print("b")
        # 4. 収束判定
        iteration +=1
        new_likehood = calc_likelihood(X, means, sigmas, pi,K)
        print(round(likehood, 2))
        print(round(new_likehood,2))



        if  (likehood - 0.1 <= new_likehood) and (new_likehood <= likehood + 0.1):
                is_converged = True
                print("%f vs %f" % (new_likehood, likehood))
        print("likehood",likehood)


        if iteration >= 20:
            is_converged = True
        likehood = new_likehood

    return gamma


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
    path = "/Users/matsumoto-hirotomo/coma/model.param.data.fast"
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
                      data_norm["beta"].tolist()],np.float)
    dblp_array = dblp_array.T
    num = 4
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
    path = "/content/drive/MyDrive/data/DBLP/model.param.data.fast"
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
    gamma = em_algorism(500,num,em,means,sigma)
    np.argmax(gamma,axis=1)
    np.save(
    "gamma{}".formata(num), # データを保存するファイル名
    gamma,  # 配列型オブジェクト（listやnp.array)
    )
    