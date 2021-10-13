# yuzu

Tired of spinning around in your chair while your feature attribution calculations slowly churn away? Unhappy with the boring names given to other approaches and want to inject more fruit into your daily conversations? Well, we have just the tool for you. Introducing Yuzu, a compressed sensing-based approach that can make in-silico saturation mutagenesis calculations on DNA, RNA, and proteins an order of magnitude faster, and enable speedy interactive exploration! Yuzu can be run in just two lines of code on any sequential PyTorch model that takes categorical sequences as input (no graph structures or multi-input/output supported yet) on either a CPU or GPU, and exhibits large speedups in both settings.

### Installation

`pip install yuzu-ism`

### How fast is it?

Across six different models, Yuzu is significantly faster than the naive ISM approach as well as another method for speeding up ISM, fastISM. 

<img src="https://github.com/kundajelab/yuzu/blob/main/figs/yuzu_timings.png" width=900>

Yuzu shines in the interactive setting, where one sequence, or a small number of sequences, are being analyzed. This setting most commonly arises when one is exploring their model in a Jupyter notebook or needs to generate visualizations of a few exemplar sequences minutes before a meeting with their advisor. Yuzu's speed comes from replacing the per-batch bookkeepping that fastISM does with the construction and decoding of probes using compressed sensing, which involves fast matrix multiplies. However, to be fair, the results suggest that when one has enough GPU memory and very large batch sizes, fastISM will be similarly fast or faster than Yuzu.

### How does it work?

Yuzu relies on two complimentary ideas: computing the difference in layer outputs between the mutated sequences and the reference sequence (the deltas), and using compressed sensing to compress these sparse deltas into a compact set of informative probes. 

<img src="https://github.com/kundajelab/yuzu/blob/main/figs/yuzu_schematic.png" width=900>

Accordingly, Yuzu begins by taking in a sparse tensor indicating where the mutations in the sequence are (A). Then, Yuzu proceeds through the layers of the network sequentially, choosing what to do next depending on the operation encountered.

(1) A convolution: Yuzu compresses the sparse deltas into a compact set of informative probes (B). Importantly, the number of probes depends only on the receptive field of the operation and not on the length of the sequence, whereas naive ISM runs each operation on a number of sequences proportional to the length of the sequence.

(2) A pool: Yuzu extracts the relevant windows surrounding the deltas, adds the reference values to the deltas, applies the operation, and subtracts the after-pool reference values to recover the deltas.

(3) An element- or position-wise operation (C): The same as (2) except the the respective operation is applied instead of the pooling operation and no extraction step is necessary. The addition and subtraction of the reference values can be done directly to the delta tensor. 

(4) A dense layer preceeded by a convolution: Although the receptive field of a dense layer spans the entire input, if the input is sparse then the sparse operation is performed to save memory and speed.

(5) A dense layer preceeded by a dense layer: Once this point is encountered, Yuzu proceeds like naive ISM.

Read the paper for more details.

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

# Line 1: Do the precomputation once or load your previously-calculated precomputation
precomputation = precompute(model, seq_len=1000, device='cpu')

# Line 2: Run yuzu_ism on as many sequences as you'd like. 
yuzu_ism, layer_timings, within_layer_timings = yuzu_ism(model, X_0, precomputation, device='cpu')
```

In only two lines of code you can significantly speed up ISM!

### Limitations

Currently, Yuzu can only be run on models of modest size (unfortunately, no Bassenji- or Enformer-scale models quite yet). This is because a limitation of the compressed sensing formulation is that all mutated sequences must be solved simultaneously. Although one likely could not fit in 6,000 sequences and a Basset model into GPU memory, remember that the mutated sequences are compressed into a much smaller set of probes. A GPU with 12GB of memory can easily handle the Yuzu procedure because Yuzu operates on the compressed probes directly. 

A second limitation is that Yuzu requires that, when the model is specified, that each operation is listed sequentially in the order that it is executed in the forward pass and that no operations other than these layers are performed in the forward pass. This is because Yuzu sequentially iterates over the layers of the model using the `model.children()` function rather than through the forward function. So, if the data is resized or flattened, it must be done through a custom layer that wraps the operation. See Tutorial 3 for more details on how to apply Yuzu to pre-trained models that were not specified in this way.
