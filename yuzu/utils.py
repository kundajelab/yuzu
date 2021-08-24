# utils.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code contains utility functions that support the main functionality
of the yuzu codebase.
"""

import numpy
import torch

def perturbations(X_0):
    """Produce all edit-distance-one pertuabtions of a sequence.

    This function will take in a single one-hot encoded sequence of length N
    and return a batch of N*(n_choices-1) sequences, each containing a single 
    perturbation from the given sequence.

    Parameters
    ----------
    X_0: numpy.ndarray, shape=(1, n_choices, seq_len)
        A one-hot encoded sequence to generate all potential perturbations.

    Returns
    -------
    X: torch.Tensor, shape=((n_choices-1)*seq_len, n_choices, seq_len)
        Each single-position perturbation of seq.
    """

    _, n_choices, seq_len = X_0.shape
    idxs = X_0[0].argmax(axis=0)

    X = numpy.tile(X_0, reps=(seq_len*(n_choices-1), 1, 1))
    for i in range(seq_len):
        for j in range(1, n_choices):
            idx = i*(n_choices-1) + (j-1)

            X[idx, idxs[i], i] = 0
            X[idx, (idxs[i]+j) % n_choices, i] = 1
        
    return torch.from_numpy(X)

def delta_perturbations(X_0):
    """Produce the deltas of all edit-distance-one perturbations of a sequence.

    This function is similar to the `perturbation` function except that, rather
    than returning the entire sequence, it returns a compressed version of the
    deltas, i.e., where the sequence is changing.

    Parameters
    ----------
    X_0: numpy.ndarray, shape=(1, n_choices, seq_len)
        A one-hot encoded sequence to generate all potential perturbations.

    Returns
    -------
    X: torch.Tensor, shape=((n_choices-1)*seq_len, n_choices, seq_len)
        Each single-position perturbation of seq.
    """

    _, n_choices, seq_len = X_0.shape
    idxs = X_0[0].argmax(axis=0)

    X = numpy.zeros((n_choices-1, n_choices, seq_len), dtype='float32')
    for i in range(seq_len):
        for j in range(n_choices-1):
            X[j, idxs[i], i] = -1
            X[j, (idxs[i]+j+1) % n_choices, i] = 1

    return torch.from_numpy(X)

def calculate_flanks(seq_len, receptive_field):
    """A helper function to calculate the flanking regions.

    This function will return the flanking regions given a receptive field and
    a sequence length. The flanking regions are those where the receptive field
    falls off the side of the sequence length which makes normal tensor
    operations not work. Instead, we have to handle these positions one at a
    time.

    Parameters
    ----------
    seq_len: int
        The length of the sequence we are calculating the flanking regions for.

    receptive_field: int
        The receptive field of the model.

    Returns
    -------
    fl: int
        The flanking region on the left side.

    fr: int
        The flanking region on the right side.

    idxs: list
        The position indexes of the flanking regions.
    """

    fl = receptive_field // 2
    fr = receptive_field - fl - receptive_field % 2
    idxs = list(it.chain(range(fl), range(seq_len-fr, seq_len)))

    return fl, fr, idxs

def tensorflow_to_pytorch(tf_model, torch_model):
    """Copy the weights from a Tensorflow model to a PyTorch model.

    This function will take in a Tensorflow model and a PyTorch model
    with the same architecture and will transfer the weights from the
    TensorFlow model to the PyTorch model. It will disable additional
    functionality provided by PyTorch or Tensorflow to ensure that the
    models are equivalent.

    Parameters
    ----------
    tf_model: tf.keras.Model
        A model implemented in Tensorflow

    torch_model: torch.nn.Module
        A model implemented in PyTorch

    Returns
    -------
    torch_model: torch.nn.Module
        A model implemented in PyTorch with the weights transferred
        from the Tensorflow model.
    """

    tf_use_layers = (tf.keras.layers.Conv1D, 
                    tf.keras.layers.BatchNormalization, 
                    tf.keras.layers.Dense)

    torch_use_layers = (torch.nn.Conv1d,
                       torch.nn.BatchNorm1d,
                       torch.nn.Linear)

    model2_layers = list(torch_model.children())

    with torch.no_grad():
        i, j = -1, -1
        while i < len(tf_model.layers) - 1 and j < len(model2_layers) - 1:
            while i < len(tf_model.layers) - 1:
                i += 1
                if isinstance(tf_model.layers[i], tf_use_layers):
                    break

            while j < len(model2_layers) - 1:
                j += 1
                if isinstance(model2_layers[j], torch_use_layers):
                    break

            if isinstance(tf_model.layers[i], tf_use_layers[0]):
                weight = numpy.array(tf_model.layers[i].weights[0])
                weight = weight.transpose(2, 1, 0)
                bias = numpy.array(tf_model.layers[i].weights[1])

                model2_layers[j].weight[:] = torch.tensor(weight)
                model2_layers[j].bias[:] = torch.tensor(bias)

            elif isinstance(tf_model.layers[i], tf_use_layers[1]):
                mu = numpy.array(tf_model.layers[i].weights[2])
                sigma = numpy.array(tf_model.layers[i].weights[3])

                model2_layers[j].affine = False

                model2_layers[j].eps = tf_model.layers[i].epsilon
                model2_layers[j].running_mean[:] = torch.tensor(mu)
                model2_layers[j].running_var[:] = torch.tensor(sigma)
            
            elif isinstance(tf_model.layers[i], tf_use_layers[2]):
                weight = numpy.array(tf_model.layers[i].weights[0])
                bias = numpy.array(tf_model.layers[i].weights[1])

                model2_layers[j].weight[:] = torch.tensor(weight.T)
                model2_layers[j].bias[:] = torch.tensor(bias)

    return torch_model
