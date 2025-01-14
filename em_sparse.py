import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import joblib 


def init_mixing_param(K):
    """混合率の初期化"""
    return np.random.dirichlet([1] * K)


def calc_gaussian_prob(x, mean, sigma):
    """多次元ガウス分布の確率を計算"""
    d = x - mean
    sigma += np.eye(sigma.shape[0]) * 1e-5  # 数値安定性を確保
    det = np.linalg.det(sigma)

    # 行列の特異性チェック
    if det <= 1e-10:  # 行列式がゼロに近い場合、修正
        sigma += np.eye(sigma.shape[0]) * 1e-3
        det = np.linalg.det(sigma)

    inv_sigma = np.linalg.inv(sigma).astype("float32")

    # ガウス分布の確率をログスケールで計算
    exp_term = -0.5 * np.dot(d.T, np.dot(inv_sigma, d))
    log_prob = exp_term - 0.5 * (np.log(det) + sigma.shape[0] * np.log(2 * np.pi))


    #return np.exp(exp_term) / np.sqrt((2 * np.pi) ** sigma.shape[0] * det)
    return np.exp(log_prob)


def calc_likelihood(X, means, sigmas, pi, K):
    """データXの現在のパラメタにおける対数尤度を計算"""
    likelihood = 0.0
    for n in X:
        temp = sum(pi[k] * calc_gaussian_prob(n, means[k], sigmas[k]) for k in range(K))
        likelihood += np.log(temp)
    return likelihood


def em_algorithm(N, K, X, means, sigmas):
    """EMアルゴリズム"""
    pi = init_mixing_param(K)
    print("pi",pi.shape)
    print(X.shape)
  
    likelihood = calc_likelihood(X, means, sigmas, pi, K)
    gamma = np.zeros((N, K))
    is_converged = False
    iteration = 0

    while not is_converged:
        # E-Step
        for n, x in enumerate(X):
            denominator = sum(pi[k] * calc_gaussian_prob(x, means[k], sigmas[k]) for k in range(K))
            print("----------------",[pi[k] * calc_gaussian_prob(x, means[k], sigmas[k]) / denominator for k in range(K)])
            gamma[n] = [pi[k] * calc_gaussian_prob(x, means[k], sigmas[k]) / denominator for k in range(K)]

        # M-Step
        Nks = gamma.sum(axis=0)
        for k in range(K):
            means[k] = np.sum(gamma[:, k, np.newaxis] * X, axis=0) / Nks[k]
            diff = X - means[k]
            sigmas[k] = (gamma[:, k] * diff.T @ diff) / Nks[k]
            pi[k] = Nks[k] / N

        # 収束判定
        new_likelihood = calc_likelihood(X, means, sigmas, pi, K)
        print(abs(new_likelihood - likelihood))
        if abs(new_likelihood - likelihood) < 0.01 or iteration >= 20:
            is_converged = True

        likelihood = new_likelihood
        iteration += 1

    return gamma, means, sigmas


def tolist(data):
    """データをリスト形式で読み込む"""
    alpha, beta, gamma = [], [], []
    with open(data, "r") as f:
        for line in f:
            a, b, g = map(float, line.strip().split(","))
            alpha.append(a)
            beta.append(b)
            gamma.append(g)
    return alpha, beta, gamma


if __name__ == "__main__":
    data_name = "Twitter"
    persona_list = {"DBLP": [5, 25, 50], "NIPS": [3, 5, 8, 12, 16], "Twitter": [5], "Reddit": [5, 20, 50, 100, 200]}
    for k in persona_list[data_name]:
        path = f"optimize/complete/{data_name}/model.param.data.fast"
        alpha, beta, gamma = tolist(path)

        # データフレーム化と標準化
        data = pd.DataFrame({"alpha": alpha, "beta": beta, "gamma": gamma})
        scaler = StandardScaler().fit(data)
        norm_data = scaler.transform(data)
        norm_df = pd.DataFrame(norm_data, columns=["alpha", "beta", "gamma"])

        # K-means クラスタリング
        pred = KMeans(n_clusters=k).fit_predict(norm_df)
        norm_df["cluster_id"] = pred

        # 初期化
        em_data = norm_df[["alpha", "beta", "gamma"]].to_numpy()
        means = [em_data[norm_df["cluster_id"] == i].mean(axis=0) for i in range(k)]
        sigmas = [np.cov(em_data[norm_df["cluster_id"] == i].T, bias=True) for i in range(k)]
        N = len(alpha)

        # EMアルゴリズム
        gamma, means, sigmas = em_algorithm(N, k, em_data, means, sigmas)
        print(means)
        print(sigmas)
        print(gamma)
        print(np.argmax(gamma,axis=1))

        # 保存
        joblib.dump(scaler, f"optimize/complete/{data_name}/persona={k}/scaler.pkl")
        np.save(f"optimize/complete/{data_name}/persona={k}/gamma.npy", gamma)
        np.save(f"optimize/complete/{data_name}/persona={k}/means.npy", means)
        np.save(f"optimize/complete/{data_name}/persona={k}/sigma.npy", sigmas)
