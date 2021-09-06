# models.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code implements several models used for baselining the implementations.
"""

import time
import numpy
import torch
import random

class Flatten(torch.nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.permute(0, 2, 1).contiguous().view(x.shape[0], -1)


class Unsqueeze(torch.nn.Module):
	def __init__(self, dim):
		super(Unsqueeze, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)


class OneLayer(torch.nn.Module):
	def __init__(self, n_inputs, n_filters=64, kernel_size=7, seq_len=None, random_state=0):
		super(OneLayer, self).__init__()
		torch.manual_seed(random_state)
		self.conv = torch.nn.Conv1d(n_inputs, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)

	def forward(self, X):
		with torch.no_grad():
			return self.conv(X)

class ToyNet(torch.nn.Module):
	def __init__(self, n_inputs, n_filters=32, kernel_size=7, seq_len=None, random_state=0):
		super(ToyNet, self).__init__()
		torch.manual_seed(random_state)
		self.conv1 = torch.nn.Conv1d(n_inputs, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)
		self.relu1 = torch.nn.ReLU()
		self.conv2 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)
		self.relu2 = torch.nn.ReLU()
		self.conv3 = torch.nn.Conv1d(n_filters, 2, kernel_size=kernel_size, padding=kernel_size // 2)

	def forward(self, X):
		with torch.no_grad():
			return self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(X)))))

class DeepSEA(torch.nn.Module):
	def __init__(self, n_inputs, seq_len=None, random_state=0):
		super(DeepSEA, self).__init__()
		torch.manual_seed(random_state)

		k = 4

		self.conv1 = torch.nn.Conv1d(4, 320, kernel_size=2*k+1, padding=k)
		self.relu1 = torch.nn.ReLU()
		self.mp1 = torch.nn.MaxPool1d(k)

		self.conv2 = torch.nn.Conv1d(320, 480, kernel_size=2*k+1, padding=k)
		self.relu2 = torch.nn.ReLU()
		self.mp2 = torch.nn.MaxPool1d(k)

		self.conv3 = torch.nn.Conv1d(480, 960, kernel_size=2*k+1, padding=k)
		self.relu3 = torch.nn.ReLU()

		self.reshape = Flatten()
		self.fc = torch.nn.Linear((seq_len // k // k) * 960, 925)
		self.sigmoid = torch.nn.Sigmoid()
		self.unsqueeze = Unsqueeze(1)

	def forward(self, X):
		with torch.no_grad():
			X = self.mp1(self.relu1(self.conv1(X)))
			X = self.mp2(self.relu2(self.conv2(X)))
			X = self.relu3(self.conv3(X))

			X = self.reshape(X)
			X = self.sigmoid(self.fc(X))
			X = self.unsqueeze(X)
			return X

class Basset(torch.nn.Module):
	def __init__(self, n_inputs, seq_len=None, random_state=0):
		super(Basset, self).__init__()
		torch.manual_seed(random_state)

		self.conv1 = torch.nn.Conv1d(4, 300, kernel_size=19, padding=9)
		self.relu1 = torch.nn.ReLU()
		self.bn1 = torch.nn.BatchNorm1d(300)
		self.maxpool1 = torch.nn.MaxPool1d(3)

		self.conv2 = torch.nn.Conv1d(300, 200, kernel_size=11, padding=5)
		self.relu2 = torch.nn.ReLU()
		self.bn2 = torch.nn.BatchNorm1d(200)
		self.maxpool2 = torch.nn.MaxPool1d(4)

		self.conv3 = torch.nn.Conv1d(200, 200, kernel_size=7, padding=3)
		self.relu3 = torch.nn.ReLU()
		self.bn3 = torch.nn.BatchNorm1d(200)
		self.maxpool3 = torch.nn.MaxPool1d(4)

		self.reshape = Flatten()

		self.fc1 = torch.nn.Linear((seq_len // 3 // 4 // 4) * 200, 1000)
		self.relu4 = torch.nn.ReLU()
		self.bn4 = torch.nn.BatchNorm1d(1000)

		self.fc2 = torch.nn.Linear(1000, 1000)
		self.relu5 = torch.nn.ReLU()
		self.bn5 = torch.nn.BatchNorm1d(1000)
		

		self.fc3 = torch.nn.Linear(1000, 164)
		self.unsqueeze = Unsqueeze(1)

	def forward(self, X):
		with torch.no_grad():
			X = self.maxpool1(self.bn1(self.relu1(self.conv1(X))))
			X = self.maxpool2(self.bn2(self.relu2(self.conv2(X))))
			X = self.maxpool3(self.bn3(self.relu3(self.conv3(X))))

			X = self.reshape(X)

			X = self.bn4(self.relu4(self.fc1(X)))
			X = self.bn5(self.relu5(self.fc2(X)))
			X = self.fc3(X)
			X = self.unsqueeze(X)
			return X

class FactorizedBasset(torch.nn.Module):
	def __init__(self, n_inputs, seq_len=None, random_state=0):
		super(FactorizedBasset, self).__init__()
		torch.manual_seed(random_state)

		# 
		self.conv11 = torch.nn.Conv1d(n_inputs, 48, kernel_size=3, padding=1)
		self.bn11 = torch.nn.BatchNorm1d(48)
		self.relu11 = torch.nn.ReLU()
		
		self.conv12 = torch.nn.Conv1d(48, 64, kernel_size=3, padding=1)
		self.bn12 = torch.nn.BatchNorm1d(64)
		self.relu12 = torch.nn.ReLU()

		self.conv13 = torch.nn.Conv1d(64, 100, kernel_size=3, padding=1)
		self.bn13 = torch.nn.BatchNorm1d(100)
		self.relu13 = torch.nn.ReLU()

		self.conv14 = torch.nn.Conv1d(100, 150, kernel_size=7, padding=3)
		self.bn14 = torch.nn.BatchNorm1d(150)
		self.relu14 = torch.nn.ReLU()

		self.conv15 = torch.nn.Conv1d(150, 300, kernel_size=7, padding=3)
		self.bn15 = torch.nn.BatchNorm1d(300)
		self.relu15 = torch.nn.ReLU()

		self.mp1 = torch.nn.MaxPool1d(3)
		#

		self.conv21 = torch.nn.Conv1d(300, 200, kernel_size=7, padding=3)
		self.bn21 = torch.nn.BatchNorm1d(200)
		self.relu21 = torch.nn.ReLU()

		self.conv22 = torch.nn.Conv1d(200, 200, kernel_size=3, padding=1)
		self.bn22 = torch.nn.BatchNorm1d(200)
		self.relu22 = torch.nn.ReLU()

		self.conv23 = torch.nn.Conv1d(200, 200, kernel_size=3, padding=1)
		self.bn23 = torch.nn.BatchNorm1d(200)
		self.relu23 = torch.nn.ReLU()

		self.mp2 = torch.nn.MaxPool1d(4)
		#

		self.conv3 = torch.nn.Conv1d(200, 200, kernel_size=7, padding=3)
		self.bn3 = torch.nn.BatchNorm1d(200)
		self.relu3 = torch.nn.ReLU()

		self.mp3 = torch.nn.MaxPool1d(4)

		self.flatten = Flatten()
		self.fc1 = torch.nn.Linear((seq_len // 3 // 4 // 4) * 200, 1000)
		self.relu4 = torch.nn.ReLU()
		self.bn4 = torch.nn.BatchNorm1d(1000)
		self.fc2 = torch.nn.Linear(1000, 1000)
		self.relu5 = torch.nn.ReLU()
		self.bn5 = torch.nn.BatchNorm1d(1000)
		self.fc3 = torch.nn.Linear(1000, 1)
		self.unsqueeze = Unsqueeze(1)

	def forward(self, x):
		with torch.no_grad():
			x = self.relu11(self.bn11(self.conv11(x)))
			x = self.relu12(self.bn12(self.conv12(x)))
			x = self.relu13(self.bn13(self.conv13(x)))
			x = self.relu14(self.bn14(self.conv14(x)))
			x = self.relu15(self.bn15(self.conv15(x)))
			x = self.mp1(x)

			x = self.relu21(self.bn21(self.conv21(x)))
			x = self.relu22(self.bn22(self.conv22(x)))
			x = self.relu23(self.bn23(self.conv23(x)))
			x = self.mp2(x)

			x = self.relu3(self.bn3(self.conv3(x)))
			x = self.mp3(x)

			x = self.flatten(x)
			x = self.bn4(self.relu4(self.fc1(x)))
			x = self.bn5(self.relu5(self.fc2(x)))
			x = self.fc3(x)
			x = self.unsqueeze(x)
			return x

class BPNet(torch.nn.Module):
	def __init__(self, n_inputs, n_filters=64, kernel_size=21, seq_len=None, n_layers=4, random_state=0):
		super(BPNet, self).__init__()
		torch.manual_seed(random_state)

		
		self.iconv = torch.nn.Conv1d(n_inputs, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()

		self.dconv1 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2, dilation=2)
		self.drelu1 = torch.nn.ReLU()

		self.dconv2 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=4, dilation=4)
		self.drelu2 = torch.nn.ReLU()        

		self.dconv3 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=8, dilation=8)
		self.drelu3 = torch.nn.ReLU()

		#self.dconv4 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=16, dilation=16)
		#self.drelu4 = torch.nn.ReLU()

		#self.dconv5 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=32, dilation=32)
		#self.drelu5 = torch.nn.ReLU()

		#self.dconv6 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=64, dilation=64)
		#self.drelu6 = torch.nn.ReLU()

		#self.dconv7 = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=128, dilation=128)
		#self.drelu7 = torch.nn.ReLU()

		self.fconv = torch.nn.Conv1d(n_filters, 1, kernel_size=75, padding=37)
		#self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

	def forward(self, X):
		with torch.no_grad():
			X = self.irelu(self.iconv(X))
			
			X = self.drelu1(self.dconv1(X))
			X = self.drelu2(self.dconv2(X))
			X = self.drelu3(self.dconv3(X))

			X = self.fconv(X)
			#X = self.logsoftmax(self.fconv(X))
			return X
