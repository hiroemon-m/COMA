# Third Party Library
import matplotlib.pyplot as plt
import numpy as np

# TODO 時間が無かったので手打ちで対応、プログラムで自動挿入出来るようにそのうち改良

fig, ax = plt.subplots(figsize=(18, 12))


# フォントサイズの調整
plt.rcParams["font.size"] = 24
# labelsizeで軸の数字の文字サイズ変更
plt.tick_params(labelsize=30)
# fontsizeで凡例の文字サイズ変更
plt.legend(fontsize=28)
ax.set_xlim(1, 5)


plt.title("Edge AUC (DBLP)", fontsize=30)

# 横軸を定義
left = np.array([1, 2, 3, 4, 5])



netevolve_data = np.load("experiment_data/baseline/complete/DBLP/attempt=0/proposed_edge_auc.npy")
m1 = netevolve_data.mean(axis=0)
std1 = netevolve_data.std(axis=0)

proposed_data = np.load("experiment_data/complete/DBLP/persona=5/proposed_edge_auc.npy")
m1_attr = proposed_data.mean(axis=0)
std1_attr = proposed_data.std(axis=0)

dual_data = np.load("result/edge/auc/dualcast_dblp.npy")
dual_m1 = dual_data.mean(axis=0)
dual_std1 = dual_data.std(axis=0)


rnn_data = np.load("result/edge/auc/rnn_dblp.npy")
rnn_m1 = rnn_data.mean(axis=0)
rnn_std1 = rnn_data.std(axis=0)

gcn_data = np.load("result/edge/auc/gcn_dblp.npy")
gcn_m1 = gcn_data.mean(axis=0)
gcn_std1 = gcn_data.std(axis=0)


vgrnn_data = np.load("result/edge/auc/vgrnn_dblp.npy")
vgrnn_m1 = vgrnn_data.mean(axis=0)
vgrnn_std1 = vgrnn_data.std(axis=0)

plt.fill_between(left, m1_attr + std1_attr, m1_attr - std1_attr, alpha=0.5)
plt.fill_between(left, m1 + std1, m1 - std1, alpha=0.5)
plt.fill_between(left, dual_m1 + dual_std1, dual_m1 - dual_std1, alpha=0.3)
plt.fill_between(left, rnn_m1 + rnn_std1, rnn_m1 - rnn_std1, alpha=0.3)
plt.fill_between(left, gcn_m1 + gcn_std1, gcn_m1 - gcn_std1, alpha=0.3)
plt.fill_between(left, vgrnn_m1 + vgrnn_std1, vgrnn_m1 - vgrnn_std1, alpha=0.3)

plt.plot(left, m1_attr, label="Proposed", lw=7)
plt.plot(left, m1, label="NetEvolve")
plt.plot(left, dual_m1, label="DualCast")
plt.plot(left, rnn_m1, label="RNN")
plt.plot(left, gcn_m1, label="GCN")
plt.plot(left, vgrnn_m1, label="VGRNN")


# 横軸を1s単位に合わせる
plt.xticks(np.arange(1, 6, 1))
# 縦軸の設定
plt.yticks(np.arange(0.75, 0.9, 0.05))
# 凡例の表示
plt.legend(
    bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=18
)

# グリッドの表示
plt.grid(
    alpha=0.8,
    linestyle="--",
)
# x 軸のラベルを設定する。
ax.set_xlabel("Time segment (year)", fontsize=24)

# y 軸のラベルを設定する。
ax.set_ylabel("Edge AUC", fontsize=24)
# 凡例の表示
plt.legend(
    bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=18
)
fig.tight_layout()
# グラフの描画
plt.show()
