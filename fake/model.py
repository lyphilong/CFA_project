#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from csv import reader 
import pandas as pd 
import time
from sklearn.model_selection import train_test_split
import cv2 
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from data import FakeDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(283024, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__=='__main__':
  trainset = FakeDataset('./fake_dataset/train/file.csv', './fake_dataset/train')
  testset = FakeDataset('./fake_dataset/test/file.csv', './fake_dataset/test')

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
  testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
  
  net = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  for epoch in range(2):  # loop over the dataset multiple times
      running_loss = 0.0
      for i, data in enumerate(trainloader):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data['image'], data['label']
          #r zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs.float())
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 1 == 0:    # print every 2000 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0

  print('Finished Training')


  PATH = './cifar_net.pth'
  torch.save(net.state_dict(), PATH)
  dataiter = iter(testloader)
  data = dataiter.next()

  correct = 0
  total = 0
  with torch.no_grad():
      for data in testloader:
          #images, labels = data
          images, labels = data['image'], data['label']
          outputs = net(images.float())
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))
