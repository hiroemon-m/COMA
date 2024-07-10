# Third Party Library
import matplotlib.pyplot as plt
import numpy as np

# TODO 時間が無かったので手打ちで対応、プログラムで自動挿入出来るようにそのうち改良
fig, ax = plt.subplots(figsize=(18, 12))

# フォントサイズの調整
plt.rcParams["font.size"] = 30
# labelsizeで軸の数字の文字サイズ変更
plt.tick_params(labelsize=24)
# fontsizeで凡例の文字サイズ変更
ax.set_xlabel("time (t)")
ax.set_xlim(1, 5)

plt.title("Attibute AUC (DBLP)", fontsize=30)

# 横軸を定義
left = np.array([1, 2, 3, 4, 5])


netevolve_data = np.load("result/attributes/auc/proposed_attr_dblp_auc.npy")
m1_attr = netevolve_data.mean(axis=0)
std1_attr = netevolve_data.std(axis=0)

proposed_data = np.load("experiment_data/DBLP/param/persona=5/proposed_attr_auc.npy")
m1 = proposed_data.mean(axis=0)
std1 = proposed_data.std(axis=0)

dual_data = np.load("result/attributes/auc/dualcast_dblp.npy")
dual_m1 = dual_data.mean(axis=0)
dual_std1 = dual_data.std(axis=0)


rnn_data = np.load("result/attributes/auc/rnn_dblp.npy")
rnn_m1 = rnn_data.mean(axis=0)
rnn_std1 = rnn_data.std(axis=0)

plt.fill_between(left, m1 + std1, m1 - std1, alpha=0.5)
plt.fill_between(left, m1_attr + std1_attr, m1_attr - std1_attr, alpha=0.5)


plt.fill_between(left, dual_m1 + dual_std1, dual_m1 - dual_std1, alpha=0.3)
plt.fill_between(left, dual_m1 + dual_std1, dual_m1 - dual_std1, alpha=0.3)
plt.fill_between(left, rnn_m1 + rnn_std1, rnn_m1 - rnn_std1, alpha=0.3)
plt.plot(left, m1, label="Proposed", lw=5)
plt.plot(left, m1_attr, label="NetEvolve", lw=5)

plt.plot(left, dual_m1, label="DualCast")
plt.plot(left, rnn_m1, label="RNN")


# 横軸を1s単位に合わせる
plt.xticks(np.arange(1, 6, 1))
# 縦軸の設定
plt.yticks(np.arange(0.5, 0.9, 0.05))

# グリッドの表示
plt.grid(
    alpha=0.8,
    linestyle="--",
)
# x 軸のラベルを設定する。
ax.set_xlabel("Time segment (year)", fontsize=24)

# y 軸のラベルを設定する。
ax.set_ylabel("Attribute AUC", fontsize=24)

# 凡例の表示
plt.legend(
    bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=18
)
fig.tight_layout()
# グラフの描画
plt.show()