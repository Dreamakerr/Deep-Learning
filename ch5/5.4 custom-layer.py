# 自定义层
# 不带参数的层
import torch
import torch.nn.Functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
      
# 数据验证
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))


# 将层作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

# 在向该网络发送随机数据后，检查均值是否为0
# 由于我们处理的是浮点数，因为存储精度的原因，我们仍然可能会看到一个非常小的非零数。
Y = net(torch.rand(4, 8))
Y.mean()


# 带参数的层
# 该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。 
# 在此实现中，我们使用修正线性单元作为激活函数。 该层需要输入参数：in_units和units，分别表示输入数和输出数。
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
      
 
# 实例化MyLinear类并访问其模型参数
linear = MyLinear(5, 3)
linear.weight


# 使用自定义层直接执行前向传播计算
linear(torch.rand(2, 5))

# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))


