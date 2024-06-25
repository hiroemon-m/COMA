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
plt.title("Edge AUC (NIPS)", fontsize=30)
ax.set_xlim(1, 5)

# 横軸を定義
left = np.array([1, 2, 3, 4, 5])

netevolove_data = np.load("result/edge/auc/proposed_edge_nips_auc.npy")
m1 = netevolove_data .mean(axis=0)
std1 = netevolove_data .std(axis=0)

proposed_data = np.load("experiment_data/NIPS/param/persona=5/proposed_edge_auc.npy")
m1_attr = proposed_data.mean(axis=0)
std1_attr = proposed_data.std(axis=0)

# DualCast
# height_dual = np.array([
#     0.9491213151927438,
#     0.9150122284563372,
#     0.8422372611464969,
#     0.8509318852239206,
#     0.8274456521739131])

dual_data = np.load("result/edge/auc/dualcast_nips.npy")
dual_m1 = dual_data.mean(axis=0)
dual_std1 = dual_data.std(axis=0)

# RNN
# height_rnn_edge = [
#     0.8890461672473867,
#     0.90409697148331,
#     0.8205944798301487,
#     0.8001303423200933,
#     0.7465277777777777,
# ]

rnn_data = np.load("result/edge/auc/rnn_edge_nips.npy")
rnn_m1 = rnn_data.mean(axis=0)
rnn_std1 = rnn_data.std(axis=0)

# GCN
height_gcn_edge = [
    0.8804917800453514,
    0.8812401093367861,
    0.8895700636942675,
    0.8779163314561544,
    0.8571428571428572,
]
gcn_data = np.load("result/edge/auc/gcn_nips.npy")
gcn_m1 = gcn_data.mean(axis=0)
gcn_std1 = gcn_data.std(axis=0)

# VGRNN
height_vgrnn_edge = [
    0.8482851473922903,
    0.8533304560494893,
    0.8558917197452229,
    0.8531442746044515,
    0.8103002070393375,
]

vgrnn_data = np.load("result/edge/auc/vgrnn_nips.npy")
vgrnn_m1 = vgrnn_data.mean(axis=0)
vgrnn_std1 = vgrnn_data.std(axis=0)

# proposed
# height_proposed = np.array([0.9571946115155991,
#                             0.9257838121474484,
#                             0.9314128943758574,
#                             0.9515861571737564,
#                             0.8522235340417159
#                             ])

plt.fill_between(left, m1_attr + std1_attr, m1_attr - std1_attr, alpha=0.5)

plt.fill_between(left, m1 + std1, m1 - std1, alpha=0.5)
plt.fill_between(left, dual_m1 + dual_std1, dual_m1 - dual_std1, alpha=0.3)
plt.fill_between(left, rnn_m1 + rnn_std1, rnn_m1 - rnn_std1, alpha=0.3)
plt.fill_between(left, gcn_m1 + gcn_std1, gcn_m1 - gcn_std1, alpha=0.3)
plt.fill_between(left, vgrnn_m1 + vgrnn_std1, vgrnn_m1 - vgrnn_std1, alpha=0.3)

# plt.plot(left, height_proposed, label="Proposed", lw=5)
plt.plot(left, m1_attr, label="Proposed", lw=5)
plt.plot(left, m1, label="Netevolve")
plt.plot(left, dual_m1, label="DualCast")
plt.plot(left, rnn_m1, label="RNN")
plt.plot(left, gcn_m1, label="GCN")
plt.plot(left, vgrnn_m1, label="VGRNN")


# 横軸を1s単位に合わせる
plt.xticks(np.arange(1, 6, 1))
# 縦軸の設定
# plt.yticks(np.arange(0.85, 1.0, 0.05))
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
