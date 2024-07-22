
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def tolist(data_path) -> None:
    alpha = []
    beta = []
    gamma = []

    with open(data_path, "r") as f:
        lines = f.readlines()      
        for index, line in enumerate(lines):
            load_datas = line[:-1].split(",")
            alpha.append(float(load_datas[0]))
            beta.append(float(load_datas[1]))
            gamma.append((float(load_datas[2])))

    return alpha, beta, gamma

def scaling(data):
    data = np.array(data).reshape(-1,1)
    sc = StandardScaler()
    sc = sc.fit(data)
    df_sc = sc.transform(data)
    print("df",df_sc)
    #平均正規化
    #df_mean = df_sc.mean() #各列の平均を返す
    #df_max = df_sc.max()
    #df_min = df_sc.min()
    #norm = (df_sc - df_mean)/(df_max - df_min)

    #minmax
    #minmax = MinMaxScaler()
    #norm = minmax.fit_transform(df_sc)

    norm = df_sc
    norm = np.array(norm).reshape(1,-1)


    return norm[0]


#pythonでは外れ値を検出すると箱ひげずをいい感じに変える
def boxplot(alpha, beta, gamma): 

    points = (alpha, beta, gamma)
    # 箱ひげ図
    fig, ax = plt.subplots()
    bp = ax.boxplot(points)
    ax.set_xticklabels(['alpha', 'beta','gamma'])

    plt.title('Box plot')
    plt.xlabel('exams')
    plt.ylabel('point')
    # Y軸のメモリのrange
    plt.ylim([-1,1])
    plt.grid()

    # 描画
    plt.show()


if __name__ == "__main__":
    data_name = "NIPS"
    path = "gamma/{}/optimized_param".format(data_name)
    alpha, beta, gamma = tolist(path)
    print(alpha)
    alpha = scaling(alpha)
    print("a",alpha)
    beta = scaling(beta)
    gamma = scaling(gamma)
    boxplot(alpha,beta,gamma)


