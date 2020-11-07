#!/usr/bin/env python
import numpy as np
from tqdm import tqdm 
from scipy import signal
import matplotlib.pyplot as plt
import os 
from pandas import DataFrame as df


class CFA_data:
  def __init__(self, N=1000, W=638, H=638):
    self.data = [] 
    self.label = []
    self.N = N
    self.W = W
    self.H = H
  
  #Create 1 img follow pattern of CFA
  #Pattern 1:RGBG 2:BGRG 3:GRGB 4:GBGR 
  def create_fake_one(self, pattern):
    img = np.random.randint(0, 255, size=(self.W, self.H))    
    checkboard = np.zeros(shape=(self.W, self.H)) 

    if pattern == 1:
      checkboard[::2, ::2] = 1
    elif pattern==2:
      checkboard[1::2, 1::2] = 1
    elif pattern==3:
      checkboard[::2, 1::2] = 1
    elif pattern==4:
      checkboard[1::2, ::2] = 1

    img = img*checkboard
    #print(img)
    return img

  def make_fake_data(self):
    print('Create fake data ...')
    for i in tqdm(range(self.N)):
      self.label.append(np.random.randint(1,5))
      self.data.append(self.create_fake_one(self.label[-1]))
    #print(self.data)
    #print(self.label)

  def apply_interpolate(self, func):
    if func==1:
      for i in range(self.N):
        self.data[i] = self.bilinear_interpolate(self.data[i])
  
  def bilinear_interpolate(self, im):
    bilinear_conv = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])
    bilinear_conv = bilinear_conv * 1/4
    return(signal.convolve2d(im, bilinear_conv, 'same'))
  
  def save(self, path='fake_dataset'):
    #if os.path.isfile(path)==True:
    #    os.mkdir('fake_dataset')
    data = df({'data': self.data, 'lablel': self.label})
    print(data.head())
    data.to_csv(os.path.join(path, 'file.csv'))
        


if  __name__=='__main__':
  data = CFA_data(N=10000, W=538, H=538);
  data.make_fake_data()
  #fig = plt.figure()
  #ax1 = fig.add_subplot(1,2,1)
  #ax1.imshow(data.data[1].astype(np.uint8), cmap='Reds')
  data.apply_interpolate(1)
  data.save()
  #ax2 = fig.add_subplot(1,2,2)
  #ax2.imshow(data.data[1].astype(np.uint8), cmap='Reds')
  #plt.show()
