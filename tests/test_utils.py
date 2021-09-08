# test_utils.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Testing Yuzu utilities.
"""

import numpy
import torch

from yuzu.utils import *

from nose.tools import assert_raises
from nose.tools import assert_equal

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

X_list = [[[0, 1, 0, 0],
	       [1, 0, 0, 0],
	       [0, 0, 1, 0],
	       [0, 0, 1, 0],
	       [0, 0, 0, 1]],

	      [[1, 0, 0, 0],
	       [1, 0, 0, 0],
	       [0, 1, 0, 0],
	       [0, 0, 1, 0],
	       [0, 0, 1, 0]]]

X_delta_perturbations = numpy.array([
	[[[0, -1, 1, 0], [0, -1, 0, 1], [1, -1, 0, 0]],
	 [[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]],
	 [[0, 0, -1, 1], [1, 0, -1, 0], [0, 1, -1, 0]],
	 [[0, 0, -1, 1], [1, 0, -1, 0], [0, 1, -1, 0]],
	 [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]]],

	[[[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]],
	 [[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]],
	 [[0, -1, 1, 0], [0, -1, 0, 1], [1, -1, 0, 0]],
	 [[0, 0, -1, 1], [1, 0, -1, 0], [0, 1, -1, 0]],
	 [[0, 0, -1, 1], [1, 0, -1, 0], [0, 1, -1, 0]]]]).transpose(0, 2, 3, 1)

X_perturbations = numpy.array([
	[[[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
	 [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]],

	[[[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
	 [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]]
	]).transpose(0, 1, 3, 2)

X_torch = torch.tensor(X_list)
X_numpy = numpy.array(X_list).transpose(0, 2, 1)


def test_perturbations_input_raises_list():
	assert_raises(ValueError, perturbations, X_list)

def test_perturbations_input_raises_torch():
	assert_raises(ValueError, perturbations, X_torch)

def test_perturbations_input_shape_small_raises():
	assert_raises(ValueError, perturbations, X_numpy[0])

def test_perturbations_input_shape_big_raises():
	assert_raises(ValueError, perturbations, [X_numpy])

def test_perturbations_shape():
	assert_equal(perturbations(X_numpy).shape, (2, 15, 4, 5))

def test_perturbations():
	assert_array_equal(perturbations(X_numpy), X_perturbations)

###

def test_delta_perturbations_input_raises_list():
	assert_raises(ValueError, delta_perturbations, X_list)

def test_delta_perturbations_input_raises_torch():
	assert_raises(ValueError, delta_perturbations, X_torch)

def test_delta_perturbations_input_shape_small_raises():
	assert_raises(ValueError, delta_perturbations, X_numpy[0])

def test_delta_perturbations_input_shape_big_raises():
	assert_raises(ValueError, delta_perturbations, [X_numpy])

def test_delta_perturbations_shape():
	assert_equal(delta_perturbations(X_numpy).shape, (2, 3, 4, 5))

def test_delta_perturbations():
	assert_array_equal(delta_perturbations(X_numpy), X_delta_perturbations)

###

def test_calculate_flanks_odd():
	assert_equal(calculate_flanks(10, 3), (1, 1, [0, 9]))
	assert_equal(calculate_flanks(25, 3), (1, 1, [0, 24]))

def test_calculate_flanks_even():
	assert_equal(calculate_flanks(10, 6), (3, 3, [0, 1, 2, 7, 8, 9]))
	assert_equal(calculate_flanks(25, 6), (3, 3, [0, 1, 2, 22, 23, 24]))