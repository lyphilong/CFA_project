#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import cv2 as cv
from tqdm import trange
import random

#Split image to array of 4 images
#Input: (2W,2H,3)
#Output: (W/2, H/2, 12)
def image_to_block(img):
	G = img[:,:,1]
	print(G.shape)
	G = G[:,:, np.newaxis]
	img = np.concatenate([img, G], axis=2)
	return np.concatenate(np.asarray([img[::2, ::2], img[::2, 1::2], img[1::2, ::2], img[1::2, 1::2]]), axis=2)



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
				idx = np.argmax(o)
				target = torch.zeros((4, Y, X), dtype=torch.long)
				target[idx] = 1
				target = target
				loss = F.nll_loss(o, target)
				return loss


def train_net(images, net, lr=1e-3, n_epochs_auxiliary=1000, n_epochs_blockwise=500, batch_size=20, block_size=32, save_path='trained_model.pt'):
	criterion = SelfLoss()
	optim = torch.optim.Adam(net.parameters(), lr=lr)
	optim.zero_grad()
	for epoch in trange(n_epochs_blockwise):
			random.shuffle(images)
			for i_img, img in enumerate(images):
					o = net(img)
					loss = criterion(o, global_best=True)
					loss.backward()
					if (i_img+1) % batch_size == 0:
							optim.step()
							optim.zero_grad()
	torch.save(net.state_dict(), save_path)




if __name__=='__main__':
	img = cv.imread('./img.png') 
	img = image_to_block(img)
	print(img.shape)
	net = Net()
	train_net(torch.Tensor(img), net)


