import numpy as np 
from sklearn.cluster import KMeans
import pandas as pd

def tolist(data) -> None:
    np_alpha = []
    np_beta = []

    with open(data, "r") as f:
        lines = f.readlines()
    
        for line in lines:
            datas = line[:-1].split(",")
            np_alpha.append(np.float32(datas[0]))
            np_beta.append(np.float32(datas[1]))

    return np_alpha,np_beta

def kmean(num,data_norm):
    dblp_array = np.array([data_norm["alpha"].tolist(),
                      data_norm["beta"].tolist()])
    dblp_array = dblp_array.T
    num = num
    pred = KMeans(n_clusters=num).fit_predict(dblp_array)
    dblp_kmean = data_norm
    dblp_kmean["cluster_id"] = pred

    data_norm.to_csv('data_norm{}.csv'.format(str(num)), index=False)

if __name__ == "__main__":
    path = "/Users/matsumoto-hirotomo/coma/model.param.data.fast"
    dblp_alpha,dblp_beta = tolist(path)
    data_dblp = pd.DataFrame({"alpha":dblp_alpha,"beta":dblp_beta})
    kmean(16,data_dblp)
    print("done")
