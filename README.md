# yuzu

Tired of spinning around in your chair while your feature attribution calculations slowly churn away? Introducing Yuzu, a compressed sensing approach that can make in-silico mutagenesis calculations on DNA, RNA, and proteins an order of magnitude faster, enabling interactive exploration! Yuzu can be run on any sequential PyTorch model (no graph structures supported yet) on either a CPU or GPU, and achieves speedups over baselines in both contexts.

### How fast is it?

### How does it work?

Yuzu relies on two complimentary components: computing only on deltas, and compressed sensing. 

(1) The first component, computing only on deltas, speeds up the process because each sequence in ISM contains only a single mutation and so the potential span of that mutation's effect on the output from each layer in the model is limited by the receptive field of that layer. For example, a convolution layer with a kernel size of 7 applied to a sequence with a single mutation in it will exhibit the same output as the original sequence outside the 13-bp window (6 bp in either direction and the central bp) will not exhibit changes in the output 7 bp in either direction from a perturbation in the input. 

(2) The second component, compressed sensing, speeds up the process by compressing these deltas from one span per mutated sequence into a small set of dense "probes." Each probe contains a Gaussian random projection of the deltas from all of the mutated sequences. Although the principles of compressed sensing are for linear models, and neural networks are notoriously non-linear, each layer before the activation is a linear operation. So, for each convolution in the model, Yuzu constructs probes from the delta matrix, runs them through a compressed convolution operation, and decodes the outputs back into a new delta matrix. When activation layers are encountered, the reference sequence is added to the delta matrix to recover the original values, the non-linear operation is applied, and the deltas are re-extracted. 


### Installation

`pip install yuzu-ism`

### Usage

Yuzu has two steps: (1) a precomputation step where statistics about a model, and the sequence length being considered, are cached, and (2) the calculation step for an example. Although the first step takes time equivalent to a traditional ISM run on a single example, it only has to be run once. Then, the second step is run on each example (or batch of examples) that you want to calculate saliency for!

```
from yuzu import precompute
from yuzu import yuzu_ism

from yuzu.models import ToyNet # An example model with three convolution layers

# Create toy sequences that are one-hot encoded
# The sequences should have shape (n_seqs, n_characters, seq_len)
idxs = numpy.random.RandomState(0).randn(3, 4, 1000).argmax(axis=1)
X_0 = numpy.zeros((3, 4, 1000), dtype='float32')
for i in range(3):
	X_0[i, idxs[i], numpy.arange(1000)] = 1

model = ToyNet(n_inputs=4, seq_len=1000)

# Do the precomputation once or load your previously-calculated precomputation
yuzu_stats = precompute(model, X_0[:1], device='cpu')

# Run yuzu_ism on as many sequences as you'd like. 
yuzu_ism, layer_timings, within_layer_timings = yuzu_ism(model, X_0, *yuzu_stats, device='cpu')
```

### Limitations

Currently, Yuzu can only be run on models of modest size (unfortunately, no Bassenji- or Enformer-scale models quite yet). This is because a limitation of the compressed sensing formulation is that all perturbed sequences must be solved simultaneously. This limitation is not as bad as it seems: remember that the perturbed sequences are compressed into a much smaller set of probes. A GPU with 12GB of memory can easily handle a Basset-sized model with a sequence length of 2,000 (6,000 sequences decoded simultaneously) because Yuzu begins by constructing the compressions directly. 
