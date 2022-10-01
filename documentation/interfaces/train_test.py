"""
Model Training and Testing
==========================

`RectiPy` provides methods for training and testing of RNNs via the `train` and `test` methods implemented on the
`rectipy.Network` class. These methods provide a wrapper to standard `torch` training and testing procedures.
Here, we will describe how you can control the behavior of these methods via various keyword arguments.

As an example, we will use a classic `reservoir computing <http://www.scholarpedia.org/article/Echo_state_network>`_
paradigm:
- We will set up a RNN of :math:`N=100` randomly and sparsely coupled LIF neurons
    (see this use example for an introduction to the LIF neuron).
- We will provide input to this RNN via an input layer of :math:`m = 4` linear neurons.
- We will project the RNN output to a fully connected readout layer.
- We will train the readout weights while keeping the RNN and input weights fixed. We will train these weights such that
    the network is capable of distinguishing different input combinations from each other.

Step 1: Network initialization
------------------------------

First, lets define our network.
"""

import numpy as np
from rectipy import Network, random_connectivity

# network parameters
N = 100
p = 0.1
node = "neuron_model_templates.spiking_neurons.lif.lif"

# generate RNN connectivity
J = random_connectivity(N, N, p, normalize=True)

# initialize rnn layer
net = Network.from_yaml(node, weights=J, source_var="s", target_var="s_in", input_var="I_ext", output_var="s",
                        spike_var="spike", spike_def="v", op="lif_op")

# %%
# In that example, we use the function :code:`random_connectivity` to generate the random coupling matrix for our RNN
# layer. Its first two arguments define the dimensionality of the matrix, whereas :code:`p` defines the sparsity of the
# coupling, i.e. :code:`p=0.1` requires that exactly :math:`n = N^2 p` entries of the matrix are non-zero entries.
# Note that by default, none of the parameters of the RNN layer are going to be optimized during training, i.e.
# automated gradient calculation is turned off. To toggle optimization on for any of these parameters, you can use the
# keyword argument :code:`train_params` as shown below:
#
# .. code-block::
#
#       Network.from_yaml(..., train_params=["weights", "tau"])
#
# Since we only want to train the readout weights in this example, we will not use this option.
# Next, we are going to add an input layer to the network.

from rectipy import input_connections

# input layer parameters
m = 4
p_in = 0.2

# generate input weights
W_in = input_connections(N, m, p_in, variance=2.0)

# add input layer
net.add_input_layer(m, weights=W_in, trainable=False)

# %%
# Here, we used the function :code:`rectipy.input_connections` to generate the coupling weights of the input layer.
# Again, the first two arguments control the dimensionality of the returned matrix, and the third argument controls the
# sparsity of the input weights. By default, the input weights are sampled from a standard Gaussian distribution,
# and :code:`variance=2.0` ensures that the variance of the Gaussian is set to :math:`\sigma^2 = 2` in this example.
# By toggling :code:`trainable`, it is possible to enable/disable optimization of the input weights during training.
#
# As a final step of the model initialization, lets add the readout layer:

# readout layer parameters
k = 3
activation_function = "softmax"

# add readout layer
net.add_output_layer(k, trainable=True, activation_function=activation_function)

# %%
# By declaring this layer as trainable, the weights of this layer are going to be optimized during training.
# We thus skipped on initializing the weights manually. Furthermore, we chose that the activation of the neurons in that
# layer are determined by the `softmax function <https://en.wikipedia.org/wiki/Softmax_function>`_,
# which is applied to the summed up synaptic inputs of each neuron in the layer.

# %%
# Step 2: Define inputs and targets
# ---------------------------------
#
#