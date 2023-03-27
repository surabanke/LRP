

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pdb

epoch_num = 15
batch_Size = 12
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


train_load = torch.utils.data.DataLoader(dataset=mnist_train, batch_size = batch_Size, shuffle = True)

def get_Activation(x, model,layers):
    features = []
    for name, layer in enumerate(model.children()):
        try:
            x = layer(x)
        except:
            break

        if str(name) in layers:
            pdb.set_trace()
            features.append(x)
            #features[layers[str(name)]] = x
    return features


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,5,5,1,0) # in, out, filter size, stride ,padding
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(5,5,2,1,0)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(5,10,2,1,0)
        self.fc1 = nn.Linear(160,10)
        #self.fc2 = nn.Linear(10,1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self,x):
        # hidden layer 1
        # 64, 1, 28, 28 (batch_size, channel, height, width)

        x = self.conv1(x) # 64, 5, 24, 24
        x = F.relu(x)
        x = self.pool1(x) # 64, 5, 12, 12

        x = self.conv2(x) # 64,5,12,12
        x = F.relu(x)
        x = self.pool2(x)
        # hidden layer 3
        x = self.conv3(x) # 64,20,4,4    
        x = torch.flatten(x,1)  
        
        x = self.fc1(x)
    
        return x

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
criterion = torch.nn.CrossEntropyLoss().to(device)

model.train()
i = 1
for epoch in range(epoch_num):
    for data, target in train_load:
        data = data.to(device) # x mnist 한장은 data[0][0]이고 사이즈는 28x28이다.
        #plt.imshow(data[0][0])
        target = target.to(device) # y = 1,4,5,6,...
        
        optimizer.zero_grad()
        output = model(data) # 12 x 10
        
        cost = criterion(output,target) # 왜 argmax해서 뽑은 class output과 비교하는게 아니라 float형 값들과 class인target값을 loss에 넣는지 이해가 안됨

        cost.backward()
        optimizer.step() 
        layers = {'0':model.conv1,'1':model.pool1, '2':model.conv2,'3':model.pool2, '4':model.conv3}

        hidden_features = get_Activation(data,model,layers)
        pdb.set_trace()

        

        if i % 1000 == 0:
            print("Train Step : {}\tLoss : {:3f}".format(i, cost.item()))
        i += 1




