# test_precompute.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Testing Yuzu precomputation capabilities.
"""

import numpy
import torch

from yuzu import precompute
from yuzu.models import DeepSEA
from yuzu.precompute import Precomputation

from nose.tools import assert_raises
from nose.tools import assert_equal
from nose.tools import assert_true

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


seq_len, n_choices = 50, 4
model = DeepSEA(n_choices, seq_len)
precomputation = precompute(model, seq_len, n_choices)

def test_precompute_output():
	assert_true(isinstance(precomputation, Precomputation))

#

def test_precomputation_output_As_len():
	assert_equal(len(precomputation.As), len(list(model.children())))

def test_precomputation_output_As_shapes():
	assert_equal(precomputation.As[0].shape, (50, 40, 3))
	assert_equal(precomputation.As[3].shape, (12, 198, 36))
	assert_equal(precomputation.As[6].shape, (3, 225, 150))

def test_precomputation_betas_len():
	assert_equal(len(precomputation.betas), len(list(model.children())))

def test_precomputation_betas_shapes():
	assert_equal(precomputation.betas[0].shape, (50, 27, 40))
	assert_equal(precomputation.betas[3].shape, (12, 132, 198))
	assert_equal(precomputation.betas[6].shape, (3, 150, 225))

def test_precomputation_masks_len():
	assert_equal(len(precomputation.masks), len(list(model.children())))

def test_precomputation_masks_shapes():
	assert_equal(len(precomputation.masks[0][0][0]), 0)
	assert_equal(precomputation.masks[0][0][1][0].shape, (150,))
	assert_equal(precomputation.masks[0][0][1][1].shape, (150,))

	assert_equal(len(precomputation.masks[0][1][0]), 8)
	assert_equal(precomputation.masks[0][1][1][0].shape, (1134,))
	assert_equal(precomputation.masks[0][1][1][1].shape, (1134,))

	assert_equal(len(precomputation.masks[3][0][0]), 2)
	assert_equal(precomputation.masks[3][0][1][0].shape, (360,))
	assert_equal(precomputation.masks[3][0][1][1].shape, (360,))

	assert_equal(len(precomputation.masks[3][1][0]), 10)
	assert_equal(precomputation.masks[3][1][1][0].shape, (264,))
	assert_equal(precomputation.masks[3][1][1][1].shape, (264,))

	assert_equal(len(precomputation.masks[6][0][0]), 2)
	assert_equal(precomputation.masks[6][0][1][0].shape, (150,))
	assert_equal(precomputation.masks[6][0][1][1].shape, (150,))

	assert_equal(len(precomputation.masks[6][1][0]), 2)
	assert_equal(precomputation.masks[6][1][1][0].shape, (150,))
	assert_equal(precomputation.masks[6][1][1][1].shape, (150,))

def test_precomputation_receptive_fields_len():
	assert_equal(len(precomputation.receptive_fields), len(list(model.children())))

def test_precomputation_receptive_fields():
	assert_equal(precomputation.receptive_fields[0], (1, 9))
	assert_equal(precomputation.receptive_fields[1], (9, 9))
	assert_equal(precomputation.receptive_fields[2], (9, 3))
	assert_equal(precomputation.receptive_fields[3], (3, 11))
	assert_equal(precomputation.receptive_fields[4], (11, 11))
	assert_equal(precomputation.receptive_fields[5], (11, 3))
	assert_equal(precomputation.receptive_fields[6], (3, 3))
	assert_equal(precomputation.receptive_fields[7], (3, 3))
	assert_equal(precomputation.receptive_fields[8], (3, 3))
	assert_equal(precomputation.receptive_fields[9], (3, 3))
	assert_equal(precomputation.receptive_fields[10], (3, 3))
	assert_equal(precomputation.receptive_fields[11], (3, 3))

def test_precomputation_n_probes_len():
	assert_equal(len(precomputation.n_probes), len(list(model.children())))

def test_precomputation_n_probes():
	assert_equal(precomputation.n_probes[0], 40)
	assert_equal(precomputation.n_probes[1], 40)
	assert_equal(precomputation.n_probes[2], 54)
	assert_equal(precomputation.n_probes[3], 198)
	assert_equal(precomputation.n_probes[4], 198)
	assert_equal(precomputation.n_probes[5], 225)
	assert_equal(precomputation.n_probes[6], 225)
	assert_equal(precomputation.n_probes[7], 225)
	assert_equal(precomputation.n_probes[8], 225)
	assert_equal(precomputation.n_probes[9], 225)
	assert_equal(precomputation.n_probes[10], 0)
	assert_equal(precomputation.n_probes[11], 0)

def test_precomputation_seq_lens_len():
	assert_equal(len(precomputation.seq_lens), len(list(model.children())))

def test_precomputation_seq_lens():
	assert_equal(precomputation.seq_lens[0], (50, 50))
	assert_equal(precomputation.seq_lens[1], (50, 50))
	assert_equal(precomputation.seq_lens[2], (50, 12))
	assert_equal(precomputation.seq_lens[3], (12, 12))
	assert_equal(precomputation.seq_lens[4], (12, 12))
	assert_equal(precomputation.seq_lens[5], (12, 3))
	assert_equal(precomputation.seq_lens[6], (3, 3))
	assert_equal(precomputation.seq_lens[7], (3, 3))
	assert_equal(precomputation.seq_lens[8], (3, 3))
	assert_equal(precomputation.seq_lens[9], (2880, 3))
	assert_equal(precomputation.seq_lens[10], (925, 3))
	assert_equal(precomputation.seq_lens[11], (925, 925))

def test_precomputation_n_nonzeros_len():
	assert_equal(len(precomputation.n_nonzeros), len(list(model.children())))

def test_precomputation_n_nonzeros():
	assert_equal(precomputation.n_nonzeros[0], (3, 27))
	assert_equal(precomputation.n_nonzeros[1], (3, 27))
	assert_equal(precomputation.n_nonzeros[2], (27, 36))
	assert_equal(precomputation.n_nonzeros[3], (36, 132))
	assert_equal(precomputation.n_nonzeros[4], (36, 132))
	assert_equal(precomputation.n_nonzeros[5], (132, 150))
	assert_equal(precomputation.n_nonzeros[6], (150, 150))
	assert_equal(precomputation.n_nonzeros[7], (150, 150))
	assert_equal(precomputation.n_nonzeros[8], (150, 150))
	assert_equal(precomputation.n_nonzeros[9], (150, 150))
	assert_equal(precomputation.n_nonzeros[10], (150, 150))
	assert_equal(precomputation.n_nonzeros[11], (150, 150))


