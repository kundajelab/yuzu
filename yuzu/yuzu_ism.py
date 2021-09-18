# yuzu_ism.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code includes the functions and utilities for the compressed sensing
ISM approach as well as the precomputation step.
"""

import time
import numpy
import torch

from .models import Flatten
from .models import Unsqueeze

from .utils import safe_to_device
from .utils import calculate_flanks
from .utils import delta_perturbations

global use_layers, ignore_layers, terminal_layers
use_layers = torch.nn.Conv1d, torch.nn.MaxPool1d, torch.nn.AvgPool1d
ignore_layers = torch.nn.ReLU, torch.nn.BatchNorm1d, torch.nn.LogSoftmax
terminal_layers = Unsqueeze, Flatten

@torch.inference_mode()
def _compressed_sensing_convolution(layer, X_delta, A, beta, mask, 
    n_probes, return_timings=False, verbose=False):
    """Calculate the output of a single layer using compressed sensing.

    This function will take a single layer, the input to that layer from
    a reference sequence, and the input to that layer from a set of sequences
    that each contain perturbations from the reference, and return the output
    of the layer for each of the sequences. Naively, this can be done by
    simply calling the layer on the sequences, i.e., `layer(X)`. This method
    will speed up the computation by using compressed sensing and a reference
    sequence to perform a significantly smaller number of calculations. 
    Specifically, rather than running `X.shape[0]` sequences through the
    layer, this approach runs `n_probes` sequences through the layer.
    
    Parameters
    ----------
    layer: torch.nn.Module
        A layer in a PyTorch model.

    X_0: torch.Tensor, shape=(1, n_filters, seq_len)
        A tensor containing the input to the layer from the reference sequence 
        that ISM is being performed on.

    X: torch.Tensor, shape=(n, n_filters, seq_len)
        A tensor containing the input to the layer for each sequence being
        evaluated.

    A: torch.Tensor, shape=(n_probes, n)
        A tensor of random Gaussian values for the  sensing matrix that 
        compresses the sequences into probes.

    betas: torch.Tensor
        A tensor containing slices of the A matrix that are ready to be used to 
        solve a regression task.

    masks: torch.Tensor 
        A tensors containing a mask of the receptive field for each example. 
        This mask is empirically derived by running the examples through the 
        network and recording where positions change with respect to the 
        reference.

    n_probes: int
        The number of probes to construct from the total number of sequences
        passed in.

    verbose: bool, optional
        Whether to print out statements showing timings and the progression
        of the computation.

    Returns
    -------
    X: torch.Tensor
        The output of the model after `X` is passed through all of the layers, 
        as if one had simply run `model(X)`. 
    """

    device = str(X_delta.device)[:4]

    if verbose:
        overall_tic = time.time()
        tic = time.time()

    bias = torch.clone(layer.bias[:])
    layer.bias[:] = 0

    n_seqs, n_nonzero, in_filters, seq_len = X_delta.shape

    # Construct the probes using the sparse contributions
    X_probe = A.matmul(X_delta.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    X_probe = X_probe.reshape(-1, in_filters, seq_len).contiguous()

    if verbose:
        if device == 'cuda': torch.cuda.synchronize()
        tic_a = time.time() - tic
        tic = time.time()

    # Run the probes, and the original sequence, through the convolution
    Y = layer(X_probe)
    Y = Y.reshape(n_seqs, -1, Y.shape[1], seq_len)
    Y = Y.permute(0, 3, 1, 2).contiguous()

    if verbose:
        if device == 'cuda': torch.cuda.synchronize()
        tic_b = time.time() - tic
        tic = time.time()

    X = torch.matmul(beta, Y).permute(0, 2, 3, 1)

    if verbose:
        if device == 'cuda': torch.cuda.synchronize()
        tic_c = time.time() - tic
        tic = time.time()

    if verbose:
        if device == 'cuda': torch.cuda.synchronize()
        final_tic = time.time() - overall_tic
        print("A:{:3.3}\tB:{:3.3}\tC:{:3.3}\tTot:{:3.3}\t".format(tic_a, tic_b, tic_c, final_tic))

    layer.bias[:] = bias

    if return_timings == True:
        t = tic_a, tic_b, tic_c, final_tic
        return X, t
    return X

@torch.inference_mode()
def _delta_pooling(layer, X_0, X_delta, masks, receptive_fields, n_probes, 
    seq_lens, n_nonzeros, n_perturbs):
    """Performs an exact pooling operation given only the deltas.

    This function will take in the reference sequence and the delta matrix and
    will perform an exact pooling layer without reconstructing the sequences 
    in their entirety. Unlike the compressed convolution layer this operation 
    is exact and accounts for instances where the kernel width of the pooling 
    operation falls outside the receptive field. This function works for any
    type of pooling layer, not just max-pooling.

    Parameters
    ----------
    layer: torch.nn.Module
        The pooling operation to be applied.

    X_0: torch.Tensor, shape=(1, n_filters, seq_lens[0])
        A tensor containing the input to the layer from the reference sequence 
        that ISM is being performed on.

    X_delta: torch.Tensor, shape=(n, n_nonzeros[0], seq_lens[0])
        A tensor containing only the deltas between the reference sequence and
        the perturbed sequences. This can be thought of as taking the values
        that are within the receptive field, a diagonal band along the original
        set of sequences, and pushing them up into a long skinny matrix.

    masks: tuple list of torch.Tensor 
        A list of tensors containing a mask of the receptive field
        for each example. This mask is empirically derived by running
        the examples through the network and recording where positions
        change with respect to the reference. The first element in the 
        tuple contains the masks before the pooling operation is applied and
        the second element contains the masks after the pooling operation
        is applied. These values are precomputed as part of the `precompute` 
        function.

    receptive_fields: tuple
        A tuple of integers indicating the maximum receptive field size
        before and after the layer is applied. The receptive field is the span 
        from the left-most sequence position to the right-most sequence 
        position that are changed in the given sequences, even if some of the 
        internal positions are not changed. These values are precomputed as
        part of the `precompute` function.
    """

    n_seqs, n_filters, _ = X_0.shape
    device = str(X_delta.device)

    seq_len0, seq_len1 = seq_lens
    rf0, rf1 = receptive_fields
    fl0, fr0, idxs0 = calculate_flanks(seq_len0, rf0)
    fl1, fr1, idxs1 = calculate_flanks(seq_len1, rf1)

    ks = layer.kernel_size
    size = int((numpy.ceil(rf0 / ks) + 1) * ks)
    step = (n_nonzeros[0] // rf0) * ks

    offset = n_nonzeros[0] - fl0 * n_nonzeros[0] // rf0 

    ##

    mask_map = numpy.maximum(numpy.arange(n_perturbs) - offset, 0) // step * ks
    mask_map2 = numpy.maximum(numpy.arange(n_perturbs) - offset, 0) // step

    row_mask0 = safe_to_device(masks[0][1][0], 'cpu')
    col_mask0 = safe_to_device(masks[0][1][1], 'cpu')
    row_mask1 = safe_to_device(masks[1][1][0], 'cpu')
    col_mask1 = safe_to_device(masks[1][1][1], 'cpu')

    row_mask00 = row_mask0 - mask_map[col_mask0]
    row_mask10 = row_mask1 - mask_map2[col_mask1]

    rows = torch.repeat_interleave(torch.arange(n_perturbs), size)
    rows = rows.reshape(n_perturbs, size)

    t_min = torch.tensor(seq_len0-1).expand(n_perturbs, size)
    cols = torch.tile(torch.arange(size), dims=(n_perturbs, 1))
    cols = torch.minimum((cols.T + mask_map[rows[:,0]]).T, t_min)

    ##

    # .permute(0, 2, 1) -> 0, 2, 1
    # .permute(1, 0, 2) -> 2, 0, 1

    X1 = X_0.unsqueeze(1).expand(-1, n_perturbs, -1, -1).permute(1, 3, 0, 2)
    X1 = X1[(rows, cols)].permute(1, 0, 2, 3)
    #X1 = X1.reshape(X1.shape[0]*X1.shape[1], X1.shape[2], X1.shape[3])

    for j, idx in enumerate(idxs0):
        r = safe_to_device(masks[0][0][j], "cpu")
        c = torch.full_like(r, idx) - mask_map[r]
        X1[(c, r)] += X_delta[:, :len(r), :, idx].permute(1, 0, 2)

    X_update = X_delta[:, :, :, fl0:seq_len0-fr0]

    # .permute(2, 0, 1).reshape(-1, X_update.shape[1])
    X_update = X_update.permute(3, 1, 0, 2).reshape(-1, n_seqs, n_filters)
    
    X1[(row_mask00, col_mask0)] += X_update

    # .permute(1, 2, 0)
    X1 = X1.permute(2, 1, 3, 0).reshape(n_perturbs*n_seqs, n_filters, -1)

    ##

    X_0 = layer(X_0)
    # .permute(2, 0, 1)
    X_1 = X_0.unsqueeze(1).expand(-1, n_perturbs, -1, -1).permute(3, 1, 0, 2)

    # .permute(2, 0, 1)
    X1 = layer(X1)
    X1 = X1.reshape(n_seqs, n_perturbs, n_filters, -1).permute(3, 1, 0, 2)
    
    X_delta_ = X1[(row_mask10, col_mask1)] - X_1[(row_mask1, col_mask1)]
    X_delta_ = X_delta_.reshape(seq_len1-fl1-fr1, n_nonzeros[1], n_seqs, n_filters)
    X_delta_ = X_delta_.permute(1, 0, 2, 3)

    X_delta = torch.zeros(n_nonzeros[1], seq_len1, n_seqs, n_filters, device=device)
    X_delta[:, fl1:seq_len1-fr1] = X_delta_

    for j, idx in enumerate(idxs1):
        r = safe_to_device(masks[1][0][j], "cpu")
        c = torch.full_like(r, idx) - mask_map2[r]
        X_delta[:len(r), idx] = X1[(c, r)] - X_0[:, :, idx].unsqueeze(0)

    X_delta = X_delta.permute(2, 0, 3, 1).contiguous()
    del X_update, X1, rows, cols
    return X_0, X_delta

@torch.inference_mode()
def _yuzu_ism(model, X_0, precomputation, device='cpu', use_layers=use_layers, 
    ignore_layers=ignore_layers, terminal_layers=terminal_layers, verbose=False):
    """Perform ISM using compressed sensing to reduce the necessary compute.

    This function will take a model, a reference sequence, and a set of
    sequences that each contain pertubations from the reference, and return
    predicted outputs from the model for each of those sequences. Naively,
    this can be done by simply calling the model on the sequences, i.e.,
    `model(X)`. This approach uses compressed sensing to leverage the
    sparsity pattern in the difference between the reference the and
    perturbed sequence at each intermediary layer to speed up computations.
    Specifically, rather than running `X.shape[0]` sequences through each
    layer, this approach runs `n_probess[i]` sequences through layer i.
    
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

    precomputation: yuzu.Precomputation
        An object containing the precomputation performed using the
        `yuzu.precompute` function.

    use_layers: tuple, optional
        The layer classes to apply compressed sensing to.

    ignore_layers: tuple, optional
        The layer classes to ignore when iterating through the layers.

    terminal_layers: tuple, optional
        The layer classes that, when encountered, cause the method to stop
        making calculations with the deltas. The calculations will return
        to the original space and layers will be applied normally from
        this point.

    verbose: bool, optional
        Whether to print out statements showing timings and the progression
        of the computation. Default is False.

    Returns
    -------
    X: torch.Tensor
        The output of the model after `X` is passed through all of the layers, 
        as if one had simply run `model(X)`. 
    """

    layer_timings, within_layer_timings = [], []

    n_seqs, n_choices, seq_len = X_0.shape
    X_idxs = X_0.argmax(axis=1)
    
    X_delta = delta_perturbations(X_0)
    X_0 = torch.from_numpy(X_0)

    model = safe_to_device(model, device)
    X_delta = safe_to_device(X_delta, device)
    X_0 = safe_to_device(X_0, device)

    n_perturbs = X_delta.shape[1] * X_delta.shape[-1]
    _, n_nonzero, in_filters, seq_len = X_delta.shape
    deltas = True

    for i, layer in enumerate(model.children()):
        tic = time.time()
        within_layer_times = None
        n_probes = precomputation.n_probes[i]

        # If it is not computationally efficient to use deltas and the deltas 
        # have already been decoded back to the full sequences, simply pass
        # those sequences through the next layer.
        # X_i -> X_i+1
        if deltas == False:
            X_0 = layer(X_0)

            X = X.reshape(X.shape[0]*X.shape[1], *X.shape[2:])
            X = layer(X)
            X = X.reshape(n_seqs, -1, *X.shape[1:])

            if verbose:
                print("A", layer, time.time() - tic)

        # If the layer is a Flatten operation, run the reference sequence
        # through the layer but ignore the deltas. We are assuming that the
        # Flatten layer will come after a convolution layer but before a
        # dense layer, and we have an efficient way for decoding those.
        elif isinstance(layer, Flatten):
            X_0 = layer(X_0)

            if verbose:
                print("F", layer, time.time() - tic)

        # If the layer is a linear layer and we still have deltas then we
        # can calculate the output of the linear layer using the sparse
        # deltas and the weight matrix of the linear layer in a fairly
        # easy manner.
        # X_conv_out -> X_dense_out
        elif isinstance(layer, torch.nn.Linear):
            X_0 = layer(X_0)

            X = torch.tile(X_0.unsqueeze(1), (1, n_perturbs, 1))

            _, _, n_filters, seq_len = X_delta.shape 
            fl, fr, idxs = calculate_flanks(seq_len, 
                precomputation.receptive_fields[i][0])

            n_out = layer.weight.shape[0]
            weight = layer.weight.reshape(n_out, seq_len, n_filters)
            weight = weight.permute(1, 2, 0).contiguous()

            # .permute(2, 0, 1)
            X_delta = X_delta.permute(0, 3, 1, 2).contiguous()
            X_update = torch.matmul(X_delta, weight)

            mask = precomputation.masks[i][0][1][1]
            mask = mask.expand(n_seqs, n_out, -1).permute(0, 2, 1)
            X.scatter_add_(1, mask, X_update[:, fl:seq_len-fr].reshape(n_seqs, -1, n_out))

            for j, idx in enumerate(idxs):
                m = precomputation.masks[i][0][0][j]
                X[:, m] += X_update[:, idx, :len(m)]

            deltas = False
            del X_delta, X_update

            if verbose:
                print("Z", layer, time.time() - tic)

        # If it previously was computationally efficient to operate on deltas
        # but is now no longer, or if you've hit a terminal layer where the
        # deltas can't be used anymore, decode the full sequences from the deltas and
        # pass them through the next layer.
        # X_dense -> X
        elif n_probes > n_perturbs or isinstance(layer, terminal_layers):
            _, n_filters, seq_len = X_0.shape
            fl, fr, idxs = calculate_flanks(seq_len, 
                precomputation.receptive_fields[i][0])

            X = torch.tile(X_0.unsqueeze(1), dims=(1, n_perturbs, 1, 1))

            for j, idx in enumerate(idxs):
                m = precomputation.masks[i][0][0][j]
                X[:, m, :, idx] += X_delta[:, :len(m), :, idx]

            X_update = X_delta[:, :, :, fl:seq_len-fr].permute(3, 1, 0, 2)
            mask = precomputation.masks[i][0][1]

            X = X.permute(3, 1, 0, 2)
            X[mask] += X_update.reshape(-1, n_seqs, n_filters)
            X = X.permute(2, 1, 3, 0)
            X = X.reshape(n_seqs*n_perturbs, n_filters, seq_len)

            #
            X_0 = layer(X_0)
            X = layer(X)

            _, n_filters, seq_len = X.shape
            X = X.reshape(n_seqs, n_perturbs, n_filters, seq_len) 

            deltas = False
            del X_delta, X_update

            if verbose:
                print("B", layer, time.time() - tic)

        # If you're still operating on the deltas but encounter a max pooling
        # layer, decode the full sequences from the deltas, pass them through
        # the max pooling layer, and recover the deltas on the other side.
        elif isinstance(layer, torch.nn.MaxPool1d):
            X_0, X_delta = _delta_pooling(layer=layer, 
                X_0=X_0, X_delta=X_delta, 
                masks=precomputation.masks[i], 
                receptive_fields=precomputation.receptive_fields[i], 
                n_probes=precomputation.n_probes[i], 
                seq_lens=precomputation.seq_lens[i], 
                n_nonzeros=precomputation.n_nonzeros[i], 
                n_perturbs=n_perturbs)

            if verbose:
                print("M", layer, time.time() - tic)

        # If it is still computationally efficient to operate on deltas
        # and the layer is a type that can be quickly processed using
        # compressed sensing then use the Yuzu-ISM procedure
        elif isinstance(layer, use_layers):
            X_0 = layer(X_0)
            X_delta = _compressed_sensing_convolution(layer=layer,
                X_delta=X_delta, A=precomputation.As[i], 
                beta=precomputation.betas[i], 
                mask=precomputation.masks[i], 
                n_probes=precomputation.n_probes[i], 
                return_timings=verbose, 
                verbose=verbose)

            if verbose:
                within_layer_times = X_delta[1]
                X_delta = X_delta[0]
                print("D", layer, time.time() - tic)

        # If it is still computationally efficient to operate on deltas
        # but the layer is a pass-through layer, e.g. activations, then
        # add the reference in, pass through the layer, and subtract
        # the reference out
        # X_dense -> X_dense
        else:
            X_delta += X_0.unsqueeze(1)
            X_delta = X_delta.reshape(-1, X_0.shape[1], X_0.shape[2])

            X_delta = layer(X_delta)
            X_0 = layer(X_0)

            X_delta = X_delta.reshape(n_seqs, -1, X_0.shape[1], X_0.shape[2])
            X_delta -= X_0.unsqueeze(1)

            if verbose:
                print("E", layer, time.time() - tic)

        within_layer_timings.append(within_layer_times)
        layer_timings.append(time.time() - tic)

    if deltas == False:
        seq_len = precomputation.seq_lens[0][0]

        Xfs = torch.square(X - X_0.unsqueeze(1)).sum(axis=-1)
        if len(Xfs.shape) == 3:
            Xfs = torch.sum(Xfs, axis=-1)
        Xfs = torch.sqrt(Xfs).reshape(n_seqs, seq_len, n_choices-1)

    else:
        ### Calculate ISM scores
        seq_len = precomputation.seq_lens[-1][1]
        fl, fr, idxs = calculate_flanks(seq_len, 
            precomputation.receptive_fields[-1][1])

        X_ism = torch.sum(torch.square(X_delta), dim=2)
        Xfs = torch.zeros(n_seqs, n_perturbs, device=device)

        mask = precomputation.masks[-1][1][1][1]
        X_ism_ = X_ism[:, :, fl:seq_len-fr]
        X_ism_ = X_ism_.permute(0, 2, 1).reshape(n_seqs, -1)
        Xfs.scatter_add_(1, mask.expand(n_seqs, -1), X_ism_)

        for i, idx in enumerate(idxs):
            m = precomputation.masks[-1][1][0][i]
            Xfs[:, m] += X_ism[:, :len(m), idx]

        seq_len = precomputation.seq_lens[0][0]
        Xfs = torch.sqrt(Xfs)
        Xfs = Xfs.reshape(n_seqs, seq_len, n_choices-1)

    seq_len = precomputation.seq_lens[0][0]
    j_idxs = torch.arange(n_seqs*seq_len)
    X_ism = torch.zeros(n_seqs*seq_len, n_choices, device=device)
    for i in range(1, n_choices):
        i_idxs = (X_idxs.flatten() + i) % n_choices
        X_ism[j_idxs, i_idxs] = Xfs[:, :, i-1].flatten()

    X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)
    X_ism = safe_to_device(X_ism, "cpu").numpy()

    #print("AFTER TIME: {:4.4} {:4.4} {:4.4}".format(time.time() - tic, toc - tic, time.time() - toc))
    return X_ism, layer_timings, within_layer_timings

def yuzu_ism(model, X_0, precomputation, batch_size=1, device='cpu', 
    use_layers=use_layers, ignore_layers=ignore_layers, 
    terminal_layers=terminal_layers, return_timings=False, verbose=False):
    """Perform ISM using compressed sensing to reduce the necessary compute.

    This function will take a model, a reference sequence, and a set of
    sequences that each contain pertubations from the reference, and return
    predicted outputs from the model for each of those sequences. Naively,
    this can be done by simply calling the model on the sequences, i.e.,
    `model(X)`. This approach uses compressed sensing to leverage the
    sparsity pattern in the difference between the reference the and
    perturbed sequence at each intermediary layer to speed up computations.
    Specifically, rather than running `X.shape[0]` sequences through each
    layer, this approach runs `n_probess[i]` sequences through layer i.
    
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

    precomputation: yuzu.Precomputation
        An object containing the precomputation performed using the
        `yuzu.precompute` function.

    batch_size: int, optional
        The number of sequences to decode at the same time. Default is 1.

    use_layers: tuple, optional
        The layer classes to apply compressed sensing to.

    ignore_layers: tuple, optional
        The layer classes to ignore when iterating through the layers.

    terminal_layers: tuple, optional
        The layer classes that, when encountered, cause the method to stop
        making calculations with the deltas. The calculations will return
        to the original space and layers will be applied normally from
        this point.

    verbose: bool, optional
        Whether to print out statements showing timings and the progression
        of the computation. Default is False.

    Returns
    -------
    X: torch.Tensor
        The output of the model after `X` is passed through all of the layers, 
        as if one had simply run `model(X)`. 
    """

    X_ism, layer_timings, within_layer_timings = [], [], []
    model = model.eval()

    starts = numpy.arange(0, len(X_0), batch_size)
    for start in starts:
        X = X_0[start:start+batch_size]
        X_ism_, layer_timings_, within_layer_timings_ = _yuzu_ism(model=model, 
            X_0=X, precomputation=precomputation, device=device,
            use_layers=use_layers, ignore_layers=ignore_layers, 
            terminal_layers=terminal_layers, verbose=verbose)

        X_ism.append(X_ism_)
        layer_timings.append(layer_timings_)
        within_layer_timings.append(within_layer_timings_)

    if return_timings:
        return numpy.vstack(X_ism), layer_timings, within_layer_timings
    return numpy.vstack(X_ism)