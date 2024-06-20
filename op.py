import numpy as np

# サンプルデータ
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]
Z = [5, 7, 9, 11, 13]

# データを配列に変換
data = np.array([X, Y, Z])

# 共分散行列を計算
cov_matrix = np.cov(data, bias=False)

print("共分散行列:")
print(cov_matrix)
