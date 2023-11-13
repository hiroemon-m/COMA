# Third Party Library
import networkx as nx
import numpy as np
import torch
from scipy import io

# First Party Library
import config

np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=)出力の数がthresholdの値より多い場合は省略される
#np.infとすることで省略しない

device = config.select_device


def spmat2sptensor(sparse_mat):
    # Third Party Library
    import torch

    dense = sparse_mat.todense()
    #疎行列を通常の行列とする
    dense = torch.from_numpy(dense.astype(np.float32)).clone().to(device)
    #numpy to Tenosrに変換しデバイス(gpu)に渡しメモリを共有させない
    return dense


def spmat2tensor(sparse_mat):
    # Third Party Library
    import torch

    shape = sparse_mat.shape
    #

    sparse_mat = sparse_mat.tocoo()
    #SciPy(scipy.sparse)の疎行列をcooにする
    sparse_tensor = torch.sparse.FloatTensor(
        torch.LongTensor([sparse_mat.row.tolist(), sparse_mat.col.tolist()]),
        torch.FloatTensor(sparse_mat.data.astype(np.float32)),
        shape,
    )
    #COO形式の疎行列の作成.インデックスの座標と値をペアになるように渡すことで初期化
    #torch.FloatTensor:32bitの浮動小数点数
    #torch.LongTensor:64bitの符号付き整数

    if torch.cuda.is_available():
        sparse_tensor = sparse_tensor.cuda()
    #GPU使えれば使う
    return sparse_tensor


class attr_graph_dynamic_spmat_DBLP:
    def __init__(self, dirIn="./data/", dataset="DBLP", T=3):
        dirIn = dirIn + dataset
        # input G
        self.T = T
        self.G_list = []
        self.A_list = []
        self.Gmat_list = []
        self.Amat_list = []
        self.Gtensor_list = []
        survive = None
        self.len = 0
        #matlabデータを読み込み
        for t in range(T):
            G_matrix = io.loadmat(
                dirIn + "/G" + str(t) + ".mat", struct_as_record=True
            )["G"]

            if survive is None:
                survive = np.array(G_matrix.sum(axis=0))
            else:
                survive = np.multiply(survive, G_matrix.sum(axis=0))

        survive = np.ravel(survive > 0)
        for t in range(T):
            G_matrix = io.loadmat(
                dirIn + "/G" + str(t) + ".mat", struct_as_record=True
            )["G"]
            A_matrix = io.loadmat(
                dirIn + "/A" + str(t) + ".mat", struct_as_record=True
            )["A"]
            G_matrix = G_matrix.T[survive].T
            A_matrix = A_matrix.T.dot(G_matrix).T
            A = nx.DiGraph()
            self.A_list.append(A)
            self.Amat_list.append(A_matrix)
            G_matrix = G_matrix.T.dot(G_matrix)
            G_matrix[G_matrix > 0] = 1.0
            G = nx.from_scipy_sparse_matrix(
                G_matrix, create_using=nx.DiGraph()
            )
            self.len = len(G.nodes())
            
            self.G_list.append(G)
            self.Gmat_list.append(G_matrix)
            self.Gtensor_list.append(spmat2tensor(G_matrix))
           
  
        #print(A.edges())
        #print(type(G_matrix[0]))
        #print(G)


class attr_graph_dynamic_spmat_NIPS:
    def __init__(self, dirIn="./data/", dataset="NIPS", T=3):
        dirIn = dirIn + dataset
        # input G
        self.T = T
        self.G_list = []
        self.A_list = []
        self.Gmat_list = []
        self.Amat_list = []
        self.Gtensor_list = []
        survive = None
        self.len = 0

        for t in range(T):
            G_matrix = io.loadmat(
                dirIn + "/G" + str(t) + ".mat", struct_as_record=True
            )["G"]
            if survive is None:
                survive = np.array(G_matrix.sum(axis=0))
            else:
                survive = np.multiply(survive, G_matrix.sum(axis=0))

        survive = np.ravel(survive > 0)
        for t in range(T):
            G_matrix = io.loadmat(
                dirIn + "/G" + str(t) + ".mat", struct_as_record=True
            )["G"]
            A_matrix = io.loadmat(
                dirIn + "/A" + str(t) + ".mat", struct_as_record=True
            )["A"]

            G_matrix = G_matrix.T[survive].T

            A_matrix = A_matrix.T.dot(G_matrix).T
            A = nx.DiGraph()
            self.A_list.append(A)
            self.Amat_list.append(A_matrix)
            G_matrix = G_matrix.T.dot(G_matrix)
            #print(G_matrix)
            G_matrix[G_matrix > 0] = 1.0
            #print(G_matrix)
            #print(G_matrix.getrow(0))
            G = nx.from_scipy_sparse_matrix(
                G_matrix, create_using=nx.DiGraph()
            )
            self.len = len(G.nodes())
            self.G_list.append(G)
            self.Gmat_list.append(G_matrix)
            self.Gtensor_list.append(spmat2tensor(G_matrix))
            
        #print(A.edges())
        #print(G_matrix)
        #print(G)
        #print(self.Gtensor_list)


class attr_graph_dynamic_spmat_twitter:
    def __init__(self, dirIn="./data/", dataset="twitter", T=1):
        n_nodes_fortime = 10000
        dirIn = dirIn + dataset
        self.T = T
        self.len = 0

        self.G_list = []
        self.A_list = []
        self.Gmat_list = []
        self.Amat_list = []
        survive = None
        for t in range(T):
            G_matrix = io.loadmat(
                dirIn + "/G" + str(t) + ".mat", struct_as_record=True
            )["G"]
            if survive is None:
                survive = np.array(G_matrix.sum(axis=0)) * 1.0 / T
            else:
                survive += np.array(G_matrix.sum(axis=0)) * 1.0 / T
        survive = np.ravel(survive > 0.1)
        for t in range(T):
            G_matrix = io.loadmat(
                dirIn + "/G" + str(t) + ".mat", struct_as_record=True
            )["G"]
            A_matrix = io.loadmat(
                dirIn + "/A" + str(t) + ".mat", struct_as_record=True
            )["A"]
            G_matrix = G_matrix[survive]
            G_matrix = G_matrix[:, survive][:n_nodes_fortime, :n_nodes_fortime]
            A_matrix = A_matrix[survive][:n_nodes_fortime, :n_nodes_fortime]
            A = nx.DiGraph()
            self.A_list.append(A)
            self.Amat_list.append(A_matrix)
            G_matrix[G_matrix > 0] = 1.0
            G = nx.from_scipy_sparse_matrix(
                G_matrix, create_using=nx.DiGraph()
            )

            self.len = len(G.nodes())
            self.G_list.append(G)
            self.Gmat_list.append(G_matrix)

        #print(self.A_list)
        #print("----")
