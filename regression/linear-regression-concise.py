import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

"""Establish a linear equation(方程）"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""Estabulish data reading function"""
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

  """Finish data read"""
batch_size = 10
data_iter = load_array((features, labels), batch_size)
"""print"""
next(iter(data_iter))

"""Introducing a neural network"""
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
"""Estabulish loss function"""
loss = MSELoss()

"""Define optimization(优化） algorithm."""
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""train"""
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        l = loss(net(features), labels)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b -b)
