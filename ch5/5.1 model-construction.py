# 回顾MLP代码
# 生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层
# 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)

# 自定义块基本功能
# 1. 将输入数据作为其前向传播函数的参数。
# 2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
# 3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
# 4. 存储和访问前向传播计算所需的参数。
# 5. 根据需要初始化模型参数。

# 下面自定义块包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。 
# 注意，下面的MLP类继承了表示块的类。 
# 我们的实现只需要提供我们自己的构造函数（Python中的__init__函数）和前向传播函数。
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
      
 
# 查看函数
net = MLP()
net(X)


# 顺序块
# 关键函数
# 一种将块逐个追加到列表中的函数；
# 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
      
# 当MySequential的前向传播函数被调用时， 每个添加的块都按照它们被添加的顺序执行。 
# 现在可以使用我们的MySequential类重新实现多层感知机。
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)


# 在向前传播函数中执行代码
# 在这个FixedHiddenMLP模型中，我们实现了一个隐藏层， 其权重（self.rand_weight）在实例化时被随机初始化，之后为常量。 、
# 这个权重不是一个模型参数，因此它永远不会被反向传播更新。 然后，神经网络将这个固定层的输出通过一个全连接层。

# 注意，在返回输出之前，模型做了一些不寻常的事情： 它运行了一个while循环，在 𝐿1范数大于 1 的条件下， 
# 将输出向量除以 2，直到它满足条件为止。 最后，模型返回了X中所有项的和。 
# 注意，此操作可能不会常用于在任何实际任务中， 我们只展示如何将任意代码集成到神经网络计算的流程中。
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
      
net = FixedHiddenMLP()
net(X)


# 混合搭配各种组合块的方法
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)


# 实现一个块，它以两个块为参数，例如net1和net2，并返回前向传播中两个网络的串联输出。这也被称为平行块。
class MySeq(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module
            
    def forward(self, X):
        for block in self._modules.values():
            X += block(X)
        return X
n = MySeq()
n(X)
