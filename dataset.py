#!/usr/bin/env python
import numpy as np
from tqdm import tqdm 
from scipy import signal
import matplotlib.pyplot as plt
import os 
import csv
from pathlib import Path
from PIL import Image

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

  def make_fake_data(self, path='./fake_dataset'):
    print('Create fake data ...')
    folder = Path(path) 
    print(folder)
    if not folder.is_dir():
      os.mkdir(path)
    
    my_file_path = os.path.join(path, 'file.csv')
    my_file = Path(my_file_path)
    if my_file.is_file():
      os.remove(my_file_path)
    else:
      my_file.touch() 
        
    with open(my_file_path, 'w', newline='') as file:
      writer = csv.writer(file)
      for i in tqdm(range(self.N)):
        cfa_pattern = np.random.randint(1,5)
        data = self.create_fake_one(cfa_pattern)
        data = self.bilinear_interpolate(data)
        data = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(data)
        im.save('{}/{}.png'.format(path,i))
        writer.writerow([i, cfa_pattern])

  def bilinear_interpolate(self, im):
    bilinear_conv = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])
    bilinear_conv = bilinear_conv * 1/4
    return(signal.convolve2d(im, bilinear_conv, 'same'))
  
       


if  __name__=='__main__':
  data = CFA_data(N=int(10e2), W=538, H=538);
  data.make_fake_data()
