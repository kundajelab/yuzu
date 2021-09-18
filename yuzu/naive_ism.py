# naive_ism.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code implements the naive form of ISM for baselining purposes.
"""

import numpy
import torch

from .utils import perturbations

@torch.inference_mode()
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

    n_seqs, n_choices, seq_len = X_0.shape
    X_idxs = X_0.argmax(axis=1)

    X = perturbations(X_0)
    X_0 = torch.from_numpy(X_0)

    if device[:4] != str(next(model.parameters()).device):
        model = model.to(device)

    if device[:4] != X_0.device:
        X_0 = X_0.to(device)

    model = model.eval()
    reference = model(X_0).unsqueeze(1)

    starts = numpy.arange(0, X.shape[1], batch_size)
    isms = []
    for i in range(n_seqs):
        y = []
        for start in starts:
            X_ = X[i, start:start+batch_size]
            if device[:4] == 'cuda': 
                X_ = X_.to(device)
            
            y_ = model(X_)
            y.append(y_)
            del X_

        y = torch.cat(y)

        ism = torch.square(y - reference[i]).sum(axis=-1)
        if len(ism.shape) == 2:
            ism = ism.sum(axis=-1)
        ism = torch.sqrt(ism)
        isms.append(ism)

        if device[:4] == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    isms = torch.stack(isms)
    isms = isms.reshape(n_seqs, seq_len, n_choices-1)

    j_idxs = torch.arange(n_seqs*seq_len)
    X_ism = torch.zeros(n_seqs*seq_len, n_choices, device=device)
    for i in range(1, n_choices):
        i_idxs = (X_idxs.flatten() + i) % n_choices
        X_ism[j_idxs, i_idxs] = isms[:, :, i-1].flatten()

    X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)

    if device[:4] == 'cuda':
        X_ism = X_ism.cpu()

    X_ism = X_ism.numpy()
    return X_ism