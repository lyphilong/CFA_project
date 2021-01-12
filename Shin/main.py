#!/usr/bin/env python
import glob
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import cv2 as cv
from tqdm import trange
import random
from math import floor 




#Read list of images
#param: path include filetype
#output: np.ndarray (N,W,H,3)
def read_images(PATH):
  list_img = [img for img in glob.glob(PATH)]
  list_img = list_img[:100]
  images = []
  for idx in range(len(list_img)):
    img_name = list_img[idx]
    img = cv.imread(img_name)
    img = img[np.newaxis, :, :, :]
    #trash
    if img.shape[1]!=3280:
      img = np.moveaxis(img, 1, 2)
    images.append(img)
    print(img.shape)
  images = np.concatenate(images, axis=0)
  print(images.shape)
  return images

#Split image to array of 4 images
#Input: (2W,2H,3)
#Output: (W/2, H/2, 12)
def image_to_block(img):
  G = img[:,:,:,1]
  G = G[:,:,:, np.newaxis]
  img = np.concatenate([img, G], axis=3)
  img = np.concatenate(np.asarray([img[:, ::2, ::2], img[:, ::2, 1::2], img[:, 1::2, ::2], img[:, 1::2, 1::2]]), axis=3)
  #img = img[np.newaxis, :, :,:]
  img = np.moveaxis(img, -1, 1)
  return img

#Prepare image before feed it to model. Image to patches
#param: np.ndarray (N, W, H, 16)
#out: np.ndarray (4N, 16, W/2, H/2)
def image_to_patch(img, patch_num):
  _, _, W, H = img.shape 
  M = np.sqrt(W*H/patch_num)
  M = int(M)
  #assert floor(M)==M  

  
  img = np.asarray([img[:, :, 0:W//2, 0:H//2],
                    img[:, :, W//2:W, 0:H//2], 
                    img[:, :, 0:W//2, H//2:H], 
                    img[:, :, W//2:W, H//2:H]])
  img = np.moveaxis(img, 0, 1)
  return img

#Net
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.gr_cnv1 = nn.Conv2d(in_channels=16, out_channels=8, groups= 2, kernel_size= 1)
    self.gr_cnv2 = nn.Conv2d(in_channels=8, out_channels= 4, kernel_size= 1)
    self.gap = nn.AdaptiveAvgPool2d((1,1))
  
  def forward(self,x):
    o = self.gr_cnv1(x)
    o = self.gr_cnv2(o)
    o = self.gap(o)
    return o


class SelfLoss(nn.Module):
    """Modified version of nn.NLLLoss for blockwise training, taking into account the fact that the target can be deduced from the output."""
    def forward(self, o):
        N, C, Y, X = o.shape
        idx = torch.argmax(o)
        target = torch.zeros((4, Y, X), dtype=torch.long)
        target[idx%C] = 1
        target = target
        loss = F.nll_loss(o, target)
        return loss


def train_net(images, net, lr=1e-4, n_epochs_auxiliary=1000, n_epochs_blockwise=500, batch_size=20, block_size=32, save_path='trained_model.pt'):
  criterion = SelfLoss()
  optim = torch.optim.Adam(net.parameters(), lr=lr)
  optim.zero_grad()
  for epoch in range(n_epochs_blockwise):
      random.shuffle(images)
      for i_img, img in enumerate(images):
          o = net(img)
          loss = criterion(o)
          print(loss.detach().numpy())
          loss.backward()
          if (i_img+1) % batch_size == 0:
              optim.step()
              optim.zero_grad()
  torch.save(net.state_dict(), save_path)




if __name__=='__main__':
  PATCH_NUM = 40
  PATH = '../images/*.png'
  images = read_images(PATH)
  images = image_to_block(images)
  images = image_to_patch(images, PATCH_NUM)
  net = Net()
  images = torch.Tensor(images)
  train_net(images, net)
  print(np.argmax(net(images[0]).detach().numpy()))


