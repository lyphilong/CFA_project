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


#TODO: Read from file to batch and continue read if it need
class CFA_data:
  def __init__(self, path='./fake_dataset/file.csv'):
    self.X_train = None 
    self.X_test = None 
    self.Y_train = None
    self.Y_test = None
    self.n = None
    self.path = path
    self.reader = reader(open(path, 'r')) 

  def read_from_file(self, batch_sz=100):
    data = []
    for i in range(batch_sz):
      tmp = self.read_line()
      img = cv2.imread('./fake_dataset/{}.png'.format(i))
      #print(tmp)
      #print(img)
      print(tmp.append(img))
      data.append(tmp.append(img))
      #print(data)
    print(data[0]) 
    #data = pd.DataFrame(data, columns = ['id', 'label', 'img']) 
    #self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(data['id'], data['label'], test_size=0.20, random_state=42)
    #print(self.X.head())


  def read_line(self):
    tmp = np.array(next(self.reader))
    tmp = tmp.astype(np.int).tolist()
    return(tmp)
   

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
  data = CFA_data()
  data.read_from_file()
  net = Net()
  print(net(torch.tensor(np.array(data.X_train), dtype=torch.float64)))

