

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pdb

epoch_num = 15
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

mnist_train = datasets.MNIST(root = './data/', 
train=True,
download = True,
transform = transforms.ToTensor())

mnist_test = datasets.MNIST(root = './data/', 
train=False,
download = True,
transform = transforms.ToTensor())


train_load = torch.utils.data.DataLoader(dataset=mnist_train, batch_size = 32, shuffle = True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,5,5,1,0) # in, out, filter size, stride ,padding
        self.conv2 = nn.Conv2d(5,5,2,1,0)
        self.conv3 = nn.Conv2d(5,10,2,1,0)
        self.fc1 = nn.Linear(160,10)
        self.fc2 = nn.Linear(10,1)

    def forward(self,x):
        # hidden layer 1
        # 64, 1, 28, 28 (batch_size, channel, height, width)
        x = self.conv1(x) # 64, 5, 24, 24
        x = F.relu(x)
        x = F.max_pool2d(x,2) # 64, 5, 12, 12
        # hidden layer 2
        x = self.conv2(x) # 64,5,12,12
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        # hidden layer 3
        x = self.conv3(x) # 64,20,4,4     
        x = torch.flatten(x,1)   
        x = self.fc1(x)
        x = self.fc2(x)

        return x

        # dropout
        #fully connected layer

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
i = 1
for epoch in range(epoch_num):
    for data, target in train_load:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pdb.set_trace()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("Train Step : {}\tLoss : {:3f}".format(i, loss.item()))
        i += 1