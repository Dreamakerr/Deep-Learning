import torch
from torch import nn
from d2l import torch as d2l

# 实现池化层的正向传播
def pool2d(X, pool_size, mode='max'):
  p_h, p_w = pool_size
  Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[2] - p_w + 1))
  for i in rage(Y.shape[0]):
    for j in range(Y.shape[1]):
      if mode == 'max':
        Y[i, j] = X[i:i + p_h, j:j + p_w].max()
      elif mode = 'avg':
        Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
      return Y

# 验证二维最大汇聚层的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))

# 验证平均汇聚层
pool2d(X, (2, 2), 'avg')

# 首先构造了一个输入张量X，它有四个维度，其中样本数和通道数都是1。
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X

# 默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同。 因此，如果我们使用形状为(3, 3)的汇聚窗口，那么默认情况下，我们得到的步幅形状为(3, 3)。
pool2d = nn.MaxPool2d(3)
pool2d(X)

# 填充和步幅可以手动设定
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

# 可以设定一个任意大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度。
pool2d = nn.MaxPool2d((2, 3), padding=(0, 1), stride=(2, 3))
pool2d(X)
# 输出：tensor([[[[ 5.,  7.],
#                [13., 15.]]]])


# 在处理多通道输入数据时，汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。 这意味着汇聚层的输出通道数与输入通道数相同。 下面，我们将在通道维度上连结张量X和X + 1，以构建具有2个通道的输入。
X = torch.cat((X, X + 1), 1)
X

# 汇聚后输出通道的数量仍然是2。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)


