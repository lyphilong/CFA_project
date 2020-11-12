#!/usr/bin/env python
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

class FakeDataset(Dataset):

		def __init__(self, csv_file, root_dir, transform=None):
				"""
				Args:
						csv_file (string): Path to the csv file with annotations.
						root_dir (string): Directory with all the images.
						transform (callable, optional): Optional transform to be applied
								on a sample.
				"""
				self.label_frame = pd.read_csv(csv_file)
				self.root_dir = root_dir
				self.transform = transform

		def __len__(self):
				return len(self.label_frame)

		def __getitem__(self, idx):
				if torch.is_tensor(idx):
						idx = idx.tolist()

				img_name = os.path.join(self.root_dir, '{}.png'.format(idx))
				image = io.imread(img_name)
				label = self.label_frame.iloc[idx, 1:]
				label = np.array([label])
				label = label.astype('float').reshape(-1)
				sample = {'image': image, 'label': label}

				if self.transform:
						sample = self.transform(sample)

				return sample



if __name__=='__main__':
	data = FakeDataset('./fake_dataset/file.csv', './fake_dataset')
	for i in tqdm(range(len(data))):
		sample = data[i]
	dataloader = DataLoader(data, batch_size=32,
													shuffle=True, num_workers=0)

	for i_batch, sample_batched in enumerate(dataloader):
			print(i_batch, sample_batched['image'].size(),
								sample_batched['label'].size())
