# 延后初始化
# 第一层会立即初始化,但其他层同样是直到数据第一次通过模型传递才会初始化
# 如果输入维度比指定维度小，可以考虑使用padding填充；
# 如果输入维度比指定维度大，可以考虑用pca等降维方法，将维度降至指定维度。
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
# print(net)

# X = torch.rand(2, 20)
net(X)
print(net)
