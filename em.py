from numpy import linalg as LA
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
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

    sigma = np.matrix(sigma) +  np.eye(sigma.shape[0]) * 1e-5  # 数値安定性のための小さなノイズ
    d = x - mean
    sigma_inv = np.linalg.inv(sigma)  # 逆行列の計算
    a = np.sqrt((2 * np.pi) ** sigma.ndim * np.linalg.det(sigma))
    b = np.exp(-0.5 * (d * sigma_inv * d.T).item())  # .item() はスカラー値を取得するために使用
    return b / a


def calc_likelihood(N,X, means, sigmas, pi,K):
    """ データXの現在のパラメタにおける対数尤度を求める
    """
    likehood = 0.0
    print("S",sigmas)
    for n in range(N):
        temp = 0.0
        for k in range(K):
            temp += pi[k] * calc_gaussian_prob(X[n], means[k], sigmas[k])
        likehood += np.log(temp)
    return likehood


def em_algorithm(N, K, X, means, sigmas):
    pi = init_mixing_param(K)
    likelihood = calc_likelihood(N,X, means, sigmas, pi, K)
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
        new_likelihood = calc_likelihood(N,X, means, sigmas, pi, K)
        if abs(new_likelihood - likelihood) < 0.01 or iteration >= 20:
            is_converged = True
        print(likelihood)
        print(new_likelihood)
        likelihood = new_likelihood
        iteration += 1


    return gamma, means



def tolist(data_path) -> None:
    np_alpha = []
    np_beta = []
    np_gamma = []

    with open(data_path, "r") as f:
        lines = f.readlines()      
        for index, line in enumerate(lines):
            load_datas = line[:-1].split(",")
            np_alpha.append(np.float32(load_datas[0]))
            np_beta.append(np.float32(load_datas[1]))
            np_gamma.append(np.float32(load_datas[2]))

    return np_alpha,np_beta,np_gamma


def em(time):
    #変数の設定

    #データの読み込み
    data_name = "NIPS"
    path = "gamma/{}/model_param_time={}".format(data_name,time)
    dblp_alpha,dblp_beta,dblp_gamma = tolist(path)
    data_dblp = pd.DataFrame({"alpha":dblp_alpha,"beta":dblp_beta,"gamma":dblp_gamma})

    #ペルソナの個数
    num = 5
    N = len(dblp_alpha)
    
    #スケーリング
    #transformer = MinMaxScaler()
    transformer = StandardScaler()
    norm = transformer.fit_transform(data_dblp)
    data_norm = data_dblp.copy(deep=True)
    data_norm["alpha"] = norm[:,0]
    data_norm["beta"] = norm[:,1]
    data_norm["gamma"] = norm[:,2]
    


    #k-mean(初期値として使用)
    dblp_array = np.array([data_norm["alpha"].tolist(),data_norm["beta"].tolist(),data_norm["gamma"].tolist()])
    dblp_array = dblp_array.T
    pred = KMeans(n_clusters=num).fit_predict(dblp_array)
    dblp_kmean = data_norm.copy()
    dblp_kmean["cluster_id"] = pred
    data = data_dblp.copy()
    data = data.loc[:,["alpha","beta","gamma"]]

    data.loc[:,["alpha","beta","gamma"]] = norm

    param_time = []
    
    for i in range(len(data["alpha"].tolist())):
        param_time.append([data["alpha"][i],data["beta"][i],data["gamma"][i]])

    em = np.array(param_time)


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
        #sigma.append(np.cov(dblp_kmean[dblp_kmean["cluster_id"]==i]["alpha"],dblp_kmean[dblp_kmean["cluster_id"]==i]["beta"],dblp_kmean[dblp_kmean["cluster_id"]==i]["gamma"],bias=True))
        #sigma.append(np.cov(dblp_kmean[dblp_kmean["cluster_id"]==i]["alpha"],dblp_kmean[dblp_kmean["cluster_id"]==i]["beta"],dblp_kmean[dblp_kmean["cluster_id"]==i]["gamma"],bias=True))



    means = []
    for i in mean:
        means.append(np.array(i))
    gamma,means = em_algorithm(N,num,em,means,sigma)
    #original_meansスケーリング前
    #original_means = transformer.inverse_transform(means)
    np.argmax(gamma,axis=1)
    np.save(
    "gamma/{}/gamma{}_{}".format(data_name,num,time), # データを保存するファイル名
    gamma,  # 配列型オブジェクト（listやnp.array)
    )
    np.save(
    "gamma/{}/means{}_{}".format(data_name,num,time), # データを保存するファイル名
    means,  # 配列型オブジェクト（listやnp.array)
    )
    
    print("割り当て率の最大",np.argmax(gamma,axis=1))
    print("gamma",gamma)
    print("means",means)
    #print("orginal-mean",original_means)

if __name__ == "__main__":
    for time in range(5):
        em(time)
