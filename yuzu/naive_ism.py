# naive_ism.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code implements the naive form of ISM for baselining purposes.
"""

import numpy
import torch

from .utils import perturbations

def naive_ism(model, X_0, batch_size=128, device='cpu'):
    """In-silico mutagenesis saliency scores. 

    This function will perform in-silico mutagenesis in a naive manner, i.e.,
    where each input sequence has a single mutation in it and the entirety
    of the sequence is run through the given model. It returns the ISM score,
    which is a vector of the L2 difference between the reference sequence 
    and the perturbed sequences with one value for each output of the model.

    Parameters
    ----------
    model: torch.nn.Module
        The model to use.

    X_0: numpy.ndarray
        The one-hot encoded sequence to calculate saliency for.

    batch_size: int, optional
        The size of the batches.

    device: str, optional
        Whether to use a 'cpu' or 'gpu'.

    Returns
    -------
    X_ism: numpy.ndarray
        The saliency score for each perturbation.
    """

    _, n_choices, seq_len = X_0.shape
    idxs = X_0[0].argmax(axis=0)

    X = perturbations(X_0)
    X_0 = torch.from_numpy(X_0)

    if device[:4] != str(next(model.parameters()).device):
        model = model.to(device)

    if device[:4] != X_0.device:
        X_0 = X_0.to(device)

    starts = numpy.arange(0, len(X), batch_size)
    X_baseline = []
    for start in starts:
        X_ = X[start:start+batch_size]
        
        if device[:4] == 'cuda': 
            X_ = X_.to(device)
        
        y = model(X_)

        X_baseline.append(y)
        del X_

    X_baseline = torch.cat(X_baseline)

    reference = model(X_0)

    ism = torch.square(X_baseline - reference)
    ism = torch.sqrt(ism.sum(axis=(1, 2))).reshape(seq_len, n_choices-1)
    if device[:4] == 'cuda':
        ism = ism.cpu()

    X_ism = torch.zeros(n_choices, seq_len, device='cpu')
    for i in range(1, n_choices):
        X_ism[(idxs + i) % n_choices, numpy.arange(seq_len)] = ism[:, i-1]
    X_ism = X_ism.numpy()
    return X_ism