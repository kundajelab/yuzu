# test_yuzu_naive_equality.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Testing the yuzu ISM implementation is equivalent to the naive ISM
implementation using the built-in models. These are regression tests. 
"""

import numpy
import torch

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal

from yuzu import yuzu_ism
from yuzu import precompute

from yuzu.naive_ism import naive_ism
from yuzu.models import *

n_seqs = 2
seq_len = 150

idxs = numpy.random.RandomState(0).randn(n_seqs, 4, seq_len).argmax(axis=1)
X = numpy.zeros((n_seqs, 4, seq_len), dtype='float32')
for i in range(n_seqs):
	X[i, idxs[i], numpy.arange(seq_len)] = 1

def evaluate_model(model, X, alpha=2):
	precomputation = precompute(model, seq_len=X.shape[2], 
		n_choices=X.shape[1], alpha=alpha)

	yuzu_isms = yuzu_ism(model, X, precomputation)
	naive_isms = naive_ism(model, X)

	assert_array_almost_equal(naive_isms, yuzu_isms, 4)

def test_one_layer():
	model = OneLayer(4, seq_len)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=10)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_toynet():
	model = ToyNet(4, seq_len)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=10)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_deepsea():
	model = DeepSEA(4, seq_len)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=10)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_basset():
	model = Basset(4, seq_len)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=10)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_factorized_basset():
	model = FactorizedBasset(4, seq_len)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=10)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=0.5)

def test_bpnet():
	model = ToyNet(4, seq_len)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=10)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

###

def test_conv_relu():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.ReLU()
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_relu_mp():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_conv():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(3),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_batchnorm_mp_conv():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.BatchNorm1d(8),
		torch.nn.MaxPool1d(3),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_relu_mp_conv():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(3),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_relu_batchnorm_mp_conv():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.ReLU(),
		torch.nn.BatchNorm1d(8),
		torch.nn.MaxPool1d(3),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_mp():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(3),
		torch.nn.MaxPool1d(2)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		Flatten(),
		torch.nn.Linear(150*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_relu_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.ReLU(),
		Flatten(),
		torch.nn.Linear(150*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_batchnorm_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.BatchNorm1d(8),
		Flatten(),
		torch.nn.Linear(150*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_relu_batchnorm_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.ReLU(),
		torch.nn.BatchNorm1d(8),
		Flatten(),
		torch.nn.Linear(150*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		Flatten(),
		torch.nn.Linear(75*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_relu_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.ReLU(),
		Flatten(),
		torch.nn.Linear(75*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_batchnorm_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.BatchNorm1d(8),
		Flatten(),
		torch.nn.Linear(75*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_batchnorm_relu_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.BatchNorm1d(8),
		torch.nn.ReLU(),
		Flatten(),
		torch.nn.Linear(75*8, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_conv_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3),
		Flatten(),
		torch.nn.Linear(75*6, 5)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_conv_dense_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3),
		Flatten(),
		torch.nn.Linear(75*6, 5),
		torch.nn.Linear(5, 3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_conv_dense_relu_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3),
		Flatten(),
		torch.nn.Linear(75*6, 5),
		torch.nn.ReLU(),
		torch.nn.Linear(5, 3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_conv_dense_batchnorm_dense():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3),
		Flatten(),
		torch.nn.Linear(75*6, 5),
		torch.nn.BatchNorm1d(5),
		torch.nn.Linear(5, 3)
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)

def test_conv_mp_conv_dense_batchnorm_dense_relu():
	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, kernel_size=7, padding=3),
		torch.nn.MaxPool1d(2),
		torch.nn.Conv1d(8, 6, kernel_size=7, padding=3),
		Flatten(),
		torch.nn.Linear(75*6, 5),
		torch.nn.BatchNorm1d(5),
		torch.nn.Linear(5, 3),
		torch.nn.ReLU()
	)

	evaluate_model(model, X)
	evaluate_model(model, X, alpha=100)
	assert_raises(AssertionError, evaluate_model, model, X, alpha=1)