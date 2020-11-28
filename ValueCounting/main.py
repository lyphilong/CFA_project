#!/usr/bin/env python
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import subprocess

def get_image_cfa_patter(path):
  cmd = [ 'dcraw', '-i', '-v', path]
  output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
  img_info = output.decode('utf-8')
  img_info = img_info.split('\n')
  for ele in img_info:
    if ele.find('Filter pattern') == 0:
      return(ele.split(' ')[2].replace('/', ''))

def MAX_c(I, i, j):
  return(np.max([I.take([i-1,j], mode='clip'), I.take([i+1,j], mode='clip'), I.take([i,j-1], mode='clip'), I.take([i,j+1], mode='clip')]))

def MIN_c(I, i,j):
  return(np.min([I.take([i-1,j], mode='clip'), I.take([i+1,j], mode='clip'), I.take([i,j-1], mode='clip'), I.take([i,j+1], mode='clip')]))

def count_green(img):
  cfa_g = np.zeros((2,2))
  w, h = img.shape
  for i in range(w):
    for j in range(h):
      if (i+j) % 2 == 1:
        if MAX_c(img, i, j) >= img[i,j] and img[i,j] >= MIN_c(img, i, j):
          cfa_g[0,1] += 1
          cfa_g[1,0] += 1
        else:
          cfa_g[0,0] += 1
          cfa_g[1,1] += 1
  return(cfa_g)


if __name__=='__main__':
  img = cv2.imread('./RAW_NIKON_D70S.tiff', 0)
  #plt.imshow(img)
  #plt.show()
  print(img)
  print(get_image_cfa_patter('./RAW_NIKON_D70S.NEF'))
  print(count_green(img))


