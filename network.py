# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# %%
for t in range(10):
    loaded_data = np.load("data/NIPS/NIPS_graph_data={}.npy".format(str(t)))
    num_nodes = len(loaded_data)

    G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)  # ノードを追加

    edge = []
    for i in range(num_nodes):

        for j in range(i + 1, num_nodes):

            if loaded_data[i][j] == 1:
                edge.append((i,j))
            G.add_edges_from(edge)  # エッジを追加


    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])

    # ノードの円周上の配置を計算
    pos = nx.circular_layout(G)

    # グラフの可視化
    #nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=50, font_size=10, font_color='black', font_weight='bold')
    #plt.title("NIPS Data Time={}".format(str(t+1)))
    #plt.show()




# %%
for t in range(10):
    loaded_data = np.load("data/DBLP/DBLP_graph_data={}.npy".format(str(t)))
    num_nodes = len(loaded_data)

    G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)  # ノードを追加

    edge = []
    for i in range(num_nodes):

        for j in range(i + 1, num_nodes):

            if loaded_data[i][j] == 1:
                edge.append((i,j))
            G.add_edges_from(edge)  # エッジを追加


    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])

    plt.figure(figsize=(50, 50))
    # ノードの円周上の配置を計算
    pos = nx.circular_layout(G)

    # グラフの可視化
    #nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=50, font_size=10, font_color='black', font_weight='bold')
    #plt.title("NIPS Data Time={}".format(str(t+1)))
    #plt.show()


