from numpy import linalg as LA
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from pickle import dump



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
        print("pu",pi)
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


    return gamma, means,sigmas,pi



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


def em(data_name,persona_num):
    #変数の設定

    #データの読み込み
    path = "gamma/{}/optimized_param".format(data_name)
    dblp_alpha,dblp_beta,dblp_gamma = tolist(path)
    print("d",dblp_alpha)
    data_dblp = pd.DataFrame({"alpha":dblp_alpha,"beta":dblp_beta,"gamma":dblp_gamma})

    #ペルソナの個数
    num = persona_num
    N = len(dblp_alpha)
    
    #標準化
    #transformer = MinMaxScaler()
    sc = StandardScaler()
    sc = sc.fit(data_dblp)
    df_sc = sc.transform(data_dblp)
    #平均正規化
    #df_mean = df_sc.mean() #各列の平均を返す
    #df_max = df_sc.max()
    #df_min = df_sc.min()
    #norm = (df_sc - df_mean)/(df_max - df_min)
    norm = df_sc
    print("mean",sc.mean_)
    print(sc.var_)




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

    print("em",em)
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
    gamma,means,sigmas,pi = em_algorithm(N,num,em,means,sigma)
    #original_meansスケーリング前
    #original_means = transformer.inverse_transform(means)
    np.argmax(gamma,axis=1)
    np.save(
    "gamma/{0}/persona={1}/gamma{1}".format(data_name,num), 
    gamma,  
    )
    np.save(
    "gamma/{0}/persona={1}/means{1}".format(data_name,num),
    means, 
    )
    np.save(
    "gamma/{0}/persona={1}/sigma{1}".format(data_name,num),
    sigmas, 
    )
    np.save(
    "gamma/{0}/persona={1}/pi{1}".format(data_name,num),
    pi, 
    )
    dump(sc,open("gamma/{0}/persona={1}/norm".format(data_name,num),"wb"))



    
    print("割り当て率の最大",np.argmax(gamma,axis=1))
    print("gamma",gamma)
    print("means",means)
    #print("orginal-mean",original_means)

if __name__ == "__main__":
    data_name = "NIPS"
    for num in [5,8,12,16]:
        em(data_name,num)