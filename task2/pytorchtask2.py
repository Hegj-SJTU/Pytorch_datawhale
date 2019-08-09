# numpy实现梯度求导
import numpy as np
import time

# N训练样本集大小 D_in输入特征 D_hid隐藏层大小 D_out输出特征
N, D_in, D_hid, D_out = 64, 1000, 100, 10
# 随机生成数据集
X_train = np.random.randn(N, D_in)
Y_train = np.random.randn(N, D_out)
print(X_train)
# 初始化权重
W1 = np.random.randn(D_in, D_hid)
W2 = np.random.randn(D_hid, D_out)

# 设置学习率
lr = 1e-6
#进行训练
start = time.clock()
for i in range(500):
    # 前向传播
    hid = X_train.dot(W1)
    hid_relu = np.maximum(0, hid)
    out = hid_relu.dot(W2)

    #计算损失函数
    loss = np.square(out-Y_train).sum()
    #输出损失函数
    print(i, loss)

    #反向传播计算梯度值
    grad_out = 2.0 * (out- Y_train)
    grad_W2 = hid_relu.T.dot(grad_out)
    grad_hid_relu = grad_out.dot(W2.T)
    grad_hid = grad_hid_relu.copy()
    grad_hid[hid<0] = 0
    grad_W1 = X_train.T.dot(grad_hid)

    #更新权重
    W1 -= lr*grad_W1
    W2 -= lr*grad_W2
time_manual = time.clock()-start
# 生成测试数据集
X_test = np.random.randn(10, D_in)
Y_test = np.random.randn(10, D_out)
print(X_test)
print('输出差值：')

hid = X_test.dot(W1)
hid_relu = np.maximum(0, hid)
out = hid_relu.dot(W2)

#差值计算
dif = out-Y_test
#输出差值
print(dif)

# 使用pytorch进行梯度更新
import torch

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n, d_in, d_hid, d_out = 64, 1000, 100, 10
x_train = torch.randn(n, d_in, device=device)
y_train = torch.randn(n, d_out, device=device)

w1 = torch.randn(d_in, d_hid, device=device, requires_grad=True)
w2 = torch.randn(d_hid, d_out, device=device, requires_grad=True)

lr=1e-6
start = time.clock()
for t in range(500):
    y_pred = x_train.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred-y_train).pow(2).sum()
    print(t, loss, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= lr*w1.grad
        w2 -= lr*w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

time_gpu = time.clock()-start


n, d_in, d_hid, d_out = 64, 1000, 100, 10
x_train = torch.randn(n, d_in)
y_train = torch.randn(n, d_out)

w1 = torch.randn(d_in, d_hid, requires_grad=True)
w2 = torch.randn(d_hid, d_out, requires_grad=True)

lr=1e-6
start = time.clock()
for t in range(500):
    y_pred = x_train.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred-y_train).pow(2).sum()
    print(t, loss, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= lr*w1.grad
        w2 -= lr*w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

time_cpu = time.clock()-start

print('time_manual:', time_manual, '\t', 'time_gpu:', time_gpu, '\t', 'time_cpu:', time_cpu)

# 使用torch.nn实现两层神经网络
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        ).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')
lr = 1e-4
start = time.process_time()
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param.data -= lr*param.grad

time_nn = time.process_time() - start
print(time_nn)