import random
import torch
from d2l import torch as d2l

"""定义线性函数"""
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.1, y.shape)
    return X, y.reshape(-1, 1)
"""实例化线性函数"""
"""生成数据样本和标签"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
"""通过生成第二个特征和 标签的散点图，可以直接观察到两者之前的线性关系。"""
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)

"""读取数据集
在下面的代码中，我们[定义一个data_iter函数， 该函数接收批量大小、特征矩阵
和标签向量作为输入，生成大小为batch_size的小批量]。 每个小批量包含一组特征和标签。"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
        indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break;
    
