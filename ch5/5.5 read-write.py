# 读写文件
# 加载和保存张量
import torch
from torch import nn
from torch.nn import functional as F

X = torch.arange(4)
torch.save(x, 'x-file')
x

x2 = torch.load('x-file')
x2

# 存储一个张量列表，然后读回内存
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)


# 将模型的参数存储在一个叫做“mlp.params”的文件中
torch.save(net.state_dict(), 'mlp.params')

# 实例化了原始多层感知机模型的一个备份
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

# 验证模型参数
Y_clone = clone(X)
Y_clone == Y
