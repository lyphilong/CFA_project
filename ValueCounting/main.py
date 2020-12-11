#!/usr/bin/env python
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import subprocess
import glob


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

# 0: G, 1: R, 2: B
def pattern_decision(img):
  CFA = np.zeros([2,2])
  CFA.fill(-1)

  cfa_g = count_green(img[0])
  cfa_r = count_green(img[1])
  cfa_b = count_green(img[2])

  if (cfa_g[0,0] > cfa_g[1,1]):
    CFA[0,0], CFA[1,1] = 0,0
    cfa_r[0,0], cfa_r[1,1], cfa_b[0,0], cfa_b[1,1] = 0, 0, 0, 0
    cfa_r_diff = np.abs(cfa_r[0,1] - cfa_r[1,0])
    cfa_b_diff = np.abs(cfa_b[0,1] - cfa_b[1,0])
  else:
    CFA[0,1], CFA[1,0] = 0,0
    cfa_r[0,1], cfa_r[1,0], cfa_b[0,1], cfa_b[1,0] = 0, 0, 0, 0
    cfa_r_diff = np.abs(cfa_r[0,0] - cfa_r[1,1])
    cfa_b_diff = np.abs(cfa_b[0,0] - cfa_b[1,1])

  if cfa_r_diff >= cfa_b_diff:
    i,j = np.unravel_index(np.argmax(cfa_r, axis=None), cfa_r.shape)
    
    CFA[i,j] = 1
    CFA[np.where(CFA== -1)] = 2
  else:
    i,j = np.unravel_index(np.argmax(cfa_r, axis=None), cfa_r.shape)

    CFA[i,j] = 2
    CFA[np.where(CFA==0)] = 1
  return CFA





if __name__=='__main__':
  list_name_img = [img for img in glob.glob('./dataset/*.NEF')]
  print(list_name_img)
  for img in list_name_img:
    cmd = ['dcraw', '-T', '-q', '0~3', img]
    subprocess.run(cmd)


  list_img = [cv2.imread(img) for img in glob.glob('./dataset/*.tiff')]
  for idx in range(len(list_img)):
    img = list_img[idx]
    print(get_image_cfa_patter(list_name_img[idx]))
    print(pattern_decision(img))


