import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
#构造假数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1) #类型0， 服从均值2 标准差1 的正态分布
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1) #类型1， 服从均值-2 标准差1 的正态分布
y1 = torch.ones(100)

#数据合并
x = torch.cat((x0, x1), 0)
print(x.dtype)
x = x.type(torch.FloatTensor)
print(x.dtype)
y = torch.cat((y0, y1), 0)
print(y.dtype)
y = y.type(torch.FloatTensor)
print(y.dtype)

#画图
#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


model = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid(),
        ).cuda()


# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 开始训练
for epoch in range(10000):
    if torch.cuda.is_available():
        x_data = Variable(x).cuda()
        y_data = Variable(y).cuda()
    else:
        x_data = Variable(x)
        y_data = Variable(y)

    out = model(x_data)
    loss = criterion(out, y_data)
    print_loss = loss.data.item()
    mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
    correct = (mask == y_data).sum()  # 计算正确预测的样本个数
    acc = correct.item() / x_data.size(0)  # 计算精度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 每隔20轮打印一下当前的误差和精度
    if (epoch + 1) % 20 == 0:
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度

for name, parameter in model.named_parameters():
    print(name, parameter)

for layer in model.modules():
   if isinstance(layer, nn.Linear):
        print(layer.weight)
        print(layer.bias)
        w1 = layer.weight[0][0]
        w2 = layer.weight[0][1]
        b = layer.bias
        print(w1.item(), w2.item(), b.item())

w0 = float(w1.item())
w1 = float(w2.item())
b = float(b.item())
plot_x = np.arange(-7, 7, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.plot(plot_x, plot_y)
plt.show()
