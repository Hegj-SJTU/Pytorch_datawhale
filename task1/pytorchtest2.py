import torch
import torchvision
from torchvision import datasets, transforms

#加载MNIST手写数字数据集数据和标签
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
trainset = datasets.MNIST(root='./data', train=True,download=True,transform=transform)
#trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=20000, shuffle=True)
trainsetloader = torch.utils.data.DataLoader(trainset,batch_size=20000,shuffle=True)
testset = datasets.MNIST(root='./data', train=True,download=True, transform=transform)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=20000, shuffle=True)
#显示数据集
dataiter = iter(trainsetloader)
images, labels = dataiter.next()
import numpy as np
import matplotlib.pyplot as plt
plt.imshow(images[0].numpy().squeeze())
plt.show()
print(images.shape)
print(labels.shape)

#设计网络结构
first_in, first_out, second_out = 28*28,  128, 10
model = torch.nn.Sequential(
    torch.nn.Linear(first_in, first_out),
    torch.nn.ReLU(),
    torch.nn.Linear(first_out, second_out),
)

#设计损失函数
loss_fn = torch.nn.CrossEntropyLoss()

#设置用于自动调节神经网络参数的优化器
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#训练神经网络
for t in range(10):
    for i, one_batch in enumerate(trainsetloader,0):
        data,label = one_batch
        data[0].view(1,784)
        data = data.view(data.shape[0],-1)

        model_output = model(data)
        loss = loss_fn(model_output , label)
        if i%500 == 0:
            print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


torch.save(model,'./my_handwrite_recognize_model.pt')

#用这个神经网络解决你的问题，比如手写数字识别，输入一个图片矩阵，然后模型返回一个数字
testdataiter = iter(testsetloader)
testimages, testlabels = testdataiter.next()

img_vector = testimages[0].squeeze().view(1,-1)
# 模型返回的是一个1x10的矩阵，第几个元素值最大那就是表示模型认为当前图片是数字几
result_digit = model(img_vector)
print("该手写数字图片识别结果为：", result_digit.max(1)[1],"标签为：",testlabels[0])