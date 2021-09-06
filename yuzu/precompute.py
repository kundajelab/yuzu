# precompute.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code includes the functions and utilities for the precomputation step
as well as the class that stores the statistics and allows them to be easily
saved and retrieved.
"""

import pickle
import torch
import random

from .models import Flatten
from .utils import calculate_flanks
from .utils import perturbations

global use_layers, ignore_layers, 
use_layers = torch.nn.Conv1d, torch.nn.MaxPool1d, torch.nn.AvgPool1d
ignore_layers = torch.nn.ReLU, torch.nn.BatchNorm1d, torch.nn.LogSoftmax


class Statistics(object):
    """A container for all of the precomputed statistics for Yuzu.

    This object will store all the statistics that are calculated during the
    precompute step, as well as methods for saving and loading the statistics
    to disk so that they don't need to be redone.

    Parameters
    ----------
    As: list of torch.Tensor, shape=(n_probes, n)
        A list of tensors containing the random Gaussian values for the 
        sensing matrix that compresses the sequences into probes. Each
        layer in the model that can be sped up using this approach has
        a different A tensor. This list of tensors is precomputed as
        part of the `precompute` function.

    betas: list of torch.Tensor
        A list of tensor containing slices of the A matrix that are
        ready to be used to solve a regression task. This list of
        tensors is precomputed as part of the `precompute` function.

    masks: list of tuples of torch.Tensor 
        A list of tuples of tensors containing a mask of the receptive field
        for each example. This mask is empirically derived by running
        the examples through the network and recording where positions
        change with respect to the reference. The first tuple contains the 
        mask before the layer is applied and the second tuple contains the 
        mask after the layer is applied. Within those tuples, the first
        item is a list of the masks applied to the edges of the sequence
        and the second item is a mask to be applied to the middle of the
        sequence.

    receptive_fields: list of tuple of ints or None
        A list of tuples of integers showing the maximum receptive field
        size for each layer before (first element) and after (second element)
        applying each layer. The receptive field is the span from the left-most
        sequence position to the right-most sequence position that are
        changed in the given sequences, even if some of the internal
        positions are not changed. This list of values is precomputed
        as part of the `precompute` function.

    n_probes: list of ints or None
        A list of values showing the number of probes to construct
        for each layer that is being compressed. This list of values
        is precomputed as part of the `precompute` function.

    seq_lens: list of tuples of ints
        A list of tuples showing the length of the sequence before
        applying the layer (first element) and after applying the layer
        (second element).

    n_nonzeros: list of tuples of ints
        A list of tuples showing the maximum number of nonzero elements per
        column before applying the layer (first element) and after applying the
        layer (second element).

    """

    def __init__(self, As, betas, masks, receptive_fields, n_probes,
        seq_lens, n_nonzeroes, device):
        self.As = As
        self.betas = betas
        self.masks = masks
        self.receptive_fields = receptive_fields
        self.n_probes = n_probes
        self.seq_lens = seq_lens
        self.n_nonzeroes = n_nonzeroes
        self.device = device


    def save(self, filename):
        args = {
            'As': self.As.cpu().numpy(),
            'betas': self.betas.cpu().numpy(),
            'masks': self.masks.cpu().numpy(),

        }

        numpy.savez_compresed(filename, )





@torch.no_grad()
def precompute(model, X_0, alpha=2, use_layers=use_layers, 
    ignore_layers=ignore_layers, device='cpu', random_state=None, 
    verbose=False):
    """Precomputing properties of the model for a Yuzu-ISM run.

    This function will take in a model, a reference sequence, and a set
    of sequences that each contain perturbations from the reference, and
    return a set of tensors that encode properties of the model that can
    speed up subsequent calculation. Because one of these properties is
    the empirical receptive field, this function will take time equal to
    one full ISM run. However, it only needs to be performed once per
    model and alpha setting and the properties can be re-used for any
    subsequent reference sequence. 
    
    Parameters
    ----------
    model: torch.nn.Module
        A PyTorch model that can contain any number and types of layers.
        However, this model must be defined in a particular way. Specifically,
        all of the operations must be defined -in order- in the `__init__`
        function. No operations can be performed outside of the context of a
        layer in the forward function, such as reshaping or applying
        activation functions. See `models.py` for examples.

    X_0: torch.Tensor, shape=(1, n_filters, seq_len)
        A tensor containing the reference sequence that ISM is being
        performed on.

    X: torch.Tensor, shape=(n, n_filters, seq_len)
        A tensor containing all of the mutations that are being considered
        for ISM. In our setting, each filter is a different nucleotide
        or amino acid and each position is being mutated to each other
        choice, making `n` usually (n_filters-1)*seq_len. However, `n`
        can be any number of sequences.

    alpha: float, optional
        A multiplier on the number of nonzero values per position to specify
        the number of probes. Default is 2.

    use_layers: tuple
        A tuple of the layers that the compressed sensing algorithm should
        be applied to. These layers should only have sparse differences in
        activation when mutated sequences are run through, such as a convolution.

    ignore_layers: tuple, optional
        Layers to ignore when passing through. ReLU layers, in particular, can
        cause trouble in determining the mask when paired with max pooling layers.
        Default is an empty tuple.

    random_state: int or None, optional
        The seed to use for the random state. Use None for no random
        seed to be set. Default is None.

    verbose: bool, optional
        Whether to print out statements showing timings and the progression
        of the computation.

    Returns
    -------
    As: list of torch.Tensor, shape=(n_probes, n)
        A list of tensors containing the random Gaussian values for the 
        sensing matrix that compresses the sequences into probes. Each
        layer in the model that can be sped up using this approach has
        a different A tensor. This list of tensors is precomputed as
        part of the `precompute` function.

    betas: list of torch.Tensor
        A list of tensor containing slices of the A matrix that are
        ready to be used to solve a regression task. This list of
        tensors is precomputed as part of the `precompute` function.

    masks: list of tuples of torch.Tensor 
        A list of tuples of tensors containing a mask of the receptive field
        for each example. This mask is empirically derived by running
        the examples through the network and recording where positions
        change with respect to the reference. The first tuple contains the 
        mask before the layer is applied and the second tuple contains the 
        mask after the layer is applied. Within those tuples, the first
        item is a list of the masks applied to the edges of the sequence
        and the second item is a mask to be applied to the middle of the
        sequence.

    receptive_fields: list of tuple of ints or None
        A list of tuples of integers showing the maximum receptive field
        size for each layer before (first element) and after (second element)
        applying each layer. The receptive field is the span from the left-most
        sequence position to the right-most sequence position that are
        changed in the given sequences, even if some of the internal
        positions are not changed. This list of values is precomputed
        as part of the `precompute` function.

    n_probess: list of ints or None
        A list of values showing the number of probes to construct
        for each layer that is being compressed. This list of values
        is precomputed as part of the `precompute` function.

    seq_lens: list of tuples of ints
        A list of tuples showing the length of the sequence before
        applying the layer (first element) and after applying the layer
        (second element).

    n_nonzeros: list of tuples of ints
        A list of tuples showing the maximum number of nonzero elements per
        column before applying the layer (first element) and after applying the
        layer (second element).
    """

    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)

    As, betas, masks, receptive_fields, = [], [], [], []
    n_probess, seq_lens, n_nonzeros = [], [], []

    X = perturbations(X_0)[0]
    X_0 = torch.from_numpy(X_0)
    n_seqs, n_choices, seq_len = X.shape

    ###

    with torch.no_grad():
        for l, layer in enumerate(model.children()):
            # Replace convolution weights with a range to ensure differences
            # are observed in intermediate values between differing inputs
            # or just ones if it's an intermediary convolution to propogate
            # those differences forward.
            if isinstance(layer, torch.nn.Conv1d):
                layer = torch.nn.Conv1d(in_channels=layer.in_channels,
                    out_channels=layer.out_channels, 
                    kernel_size=layer.kernel_size, padding=layer.padding,
                    dilation=layer.dilation, groups=layer.groups, 
                    bias=False, padding_mode=layer.padding_mode)

                if l == 0:
                    w = torch.arange(layer.in_channels)
                    w = torch.tile(w, (layer.out_channels, layer.kernel_size[0], 1))
                    layer.weight[:] = w.permute(0, 2, 1)
                else:
                    layer.weight[:] = 1

            # For the purpose of empirically defining the receptive field,
            # replace max pools by average pools because max pools can be
            # effected solely by values outside the known receptive field.
            if isinstance(layer, torch.nn.MaxPool1d):
                layer = torch.nn.AvgPool1d(kernel_size=layer.kernel_size,
                    stride=layer.stride, padding=layer.padding, 
                    ceil_mode=layer.ceil_mode)

            
            # Only go through manually specified layers.
            if isinstance(layer, use_layers) and not isinstance(layer, ignore_layers):
                X_delta = torch.abs(X - X_0).max(axis=1).values
                before_mask = (X_delta >= 1).type(dtype=torch.bool)
                bn_nonzero = int(before_mask.sum(dim=0).max())
                bseq_len = X_0.shape[-1]

                b_receptive_field = 0
                rows, cols = torch.where(before_mask == True)
                for row in torch.unique(rows):
                    col_idxs = cols[rows == row]
                    field = col_idxs.max() - col_idxs.min() + 1
                    b_receptive_field = max(b_receptive_field, field.item())

                bfl, bfr, bidxs = calculate_flanks(bseq_len, b_receptive_field)

                ###
                X_0, X = layer(X_0), layer(X)

                # Ensure that the average can't go below if the kernel size
                # of the max pooling layer goes over the receptive field.
                if isinstance(layer, torch.nn.AvgPool1d):
                    X_0 = X_0 * layer.kernel_size[0]
                    X = X * layer.kernel_size[0]

                X = X - X_0
                X_0 = X_0 - X_0

                X_delta = torch.abs(X).max(axis=1).values
                mask = (X_delta >= 1).type(dtype=torch.bool)

                n_nonzero = int(mask.sum(dim=0).max())
                n_probes = int(n_nonzero * alpha)
                seq_len = X_0.shape[2]

                receptive_field = 0
                rows, cols = torch.where(mask == True)
                for row in torch.unique(rows):
                    col_idxs = cols[rows == row]
                    field = col_idxs.max() - col_idxs.min() + 1
                    receptive_field = max(receptive_field, field.item())

                fl, fr, idxs = calculate_flanks(seq_len, receptive_field)

                receptive_field_ = (b_receptive_field, receptive_field)
                seq_lens_ = (bseq_len, seq_len)
                n_nonzeros_ = (bn_nonzero, n_nonzero)

                # If the layer is a convolution layer, pre-compute the
                # random values and regression coefficients.
                if isinstance(layer, torch.nn.Conv1d):
                    A = torch.FloatTensor(n_seqs, n_probes).normal_(0, 1)
                    A = A.to(device)

                    # Calculate regression statistics for the middle of the
                    # sequences where a constant number of features are present
                    phis = torch.FloatTensor(seq_len-fl-fr, n_nonzero, n_probes)
                    phis = phis.to(device)

                    for i in range(fl, seq_len-fr):
                        phis[i-fl] = A[mask[:,i]]

                    # Calculate betas, the regression coefficients
                    beta = torch.zeros(seq_len, n_nonzero, n_probes, device=device)
                    for i, idx in enumerate(idxs):
                        m = mask[:,idx]
                        phi = A[m]
                        b = torch.linalg.inv(phi.matmul(phi.T)).matmul(phi)
                        beta[idx, :int(m.sum()), :] = b

                    grams = torch.bmm(phis, phis.permute(0, 2, 1))
                    beta[fl:seq_len-fr] = torch.linalg.inv(grams).bmm(phis)

                    # Compact alphas
                    A_ = torch.empty(seq_len, n_probes, bn_nonzero, device=device)
                    for i in range(seq_len):
                        m = before_mask[:, i]
                        A_[i, :, :int(m.sum())] = A[m].T

                    del A
                else:
                    A_, beta = False, False

                # Calculate clipped masks (ignoring edges)
                if device[:4] == 'cuda':
                    before_mask = before_mask.cuda()
                    mask = mask.cuda()

                clipped_before_mask = torch.clone(before_mask)
                clipped_before_mask[:, :bfl] = False
                clipped_before_mask[:, bseq_len-bfr:] = False

                clipped_mask = torch.clone(mask)
                clipped_mask[:, :fl] = False
                clipped_mask[:, seq_len-fr:] = False

                mask_ = [([], torch.where(clipped_before_mask.T == True)),
                         ([], torch.where(clipped_mask.T == True))]

                # Calculate masks in edges
                for i, idx in enumerate(bidxs):
                    m = torch.where(before_mask[:, idx] == True)[0]
                    mask_[0][0].append(m)

                for i, idx in enumerate(idxs):
                    m = torch.where(mask[:, idx] == True)[0]
                    mask_[1][0].append(m)

                del mask, clipped_mask, X_delta

            else:
                # If the layer shouldn't be ignored but also won't be applied
                # to the deltas and so doesn't need anything special (e.g.,
                # a flatten or unsqueeze layer) then run it through as normal.
                if not isinstance(layer, ignore_layers):
                    bseq_len = X.shape[-1]

                    #
                    X_0, X = layer(X_0), layer(X)
                    X_delta = torch.abs(X - X_0)

                    # Depending on the layer, modify the deltas and extract
                    # the sequence length.
                    if len(X_delta.shape) > 2:
                        X_delta = X_delta.max(axis=1).values
                        seq_len = X_0.shape[-1]
                    else:
                        seq_len = seq_lens_[1]

                    mask = (X_delta >= 1).type(dtype=torch.bool)
                    n_nonzero = mask.sum(dim=0).max()
                    n_probes = int(n_nonzero * alpha)

                    seq_lens_ = bseq_len, seq_len

                    del X_delta
                else:
                    seq_lens_ = seq_lens_[1], seq_lens_[1]

                A_, beta = None, None
                receptive_field_ = receptive_field_[1], receptive_field_[1]
                mask_ = mask_[1], mask_[1]

            if verbose:
                print(l, layer, n_probes, receptive_field_, seq_lens_, n_nonzeros_)

            As.append(A_)
            betas.append(beta)
            masks.append(mask_)
            receptive_fields.append(receptive_field_)
            n_probess.append(n_probes)
            seq_lens.append(seq_lens_)
            n_nonzeros.append(n_nonzeros_)

            if device[:4] == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    del X, X_0
    if device[:4] == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return As, betas, masks, receptive_fields, n_probess, seq_lens, n_nonzeros