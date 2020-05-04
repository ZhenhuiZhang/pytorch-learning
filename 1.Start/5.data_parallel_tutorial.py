import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters 和 DataLoaders(定义参数。)
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# 设备(Device）:
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

# 要制作一个虚拟(随机）数据集，只需实现__getitem__。
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

"""
简单模型
作为演示，我们的模型只接受一个输入，执行一个线性操作，然后得到结果。然而，你能在任何模型(CNN，RNN，Capsule Net等）上使用DataParallel。
我们在模型内部放置了一条打印语句来检测输入和输出向量的大小。请注意批等级为0时打印的内容。
"""
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

if __name__ == '__main__':
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                            batch_size=batch_size, shuffle=True)
    # print(rand_loader)

    model = Model(input_size, output_size)
    if torch.cuda.device_count() > 1: 
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    for data in rand_loader: 
        # print(data)
        input = data.to(device)
        print(data)
        output = model(input)
        print("Outside: input size", input.size(),
            "output_size", output.size())