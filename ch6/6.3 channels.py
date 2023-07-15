import torch
from d2l import torch as d2l

# 实现多输入通道互相关运算
def corr2d_multi_in(X, K):
  return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

# 验证
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)

# 多输出通道
def corr2d_multi_in_out(X, K):
  # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算
  # 最后将所有结果都叠加在一起
  return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 通过将核张量K与k+1和k+2连接起来，构造一个具有3个输出通道的卷积核
K = torch.stack((K, K + 1, K + 2), 0)
K.shape

corr2d_multi_in_out(X, K)

# 使用全连接层实现1×1卷积。 
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
