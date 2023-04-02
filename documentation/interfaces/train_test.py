"""
Model Training and Testing
==========================

`RectiPy` provides methods for training and testing of RNNs via the `train` and `test` methods implemented on the
`rectipy.Network` class. These methods provide a wrapper to standard `torch` training and testing procedures.
Here, we will describe how you can control the behavior of these methods via various keyword arguments.

As an example, we will use a classic `reservoir computing <http://www.scholarpedia.org/article/Echo_state_network>`_
paradigm:
- We will set up a RNN of :math:`N=500` randomly and sparsely coupled QIF neurons with spike-frequency adaptation
    (see `this use example <https://rectipy.readthedocs.io/en/latest/auto_models/qif.html>`_ for an introduction to this neuron model).
- We will provide input to this RNN via an input layer of :math:`m = 3` linear neurons.
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
N = 500
p = 0.1
node = "neuron_model_templates.spiking_neurons.qif.qif_sfa"

# generate RNN connectivity
J = random_connectivity(N, N, p, normalize=True)

# initialize rnn layer
net = Network.from_yaml(node, weights=J, source_var="s", target_var="s_in", input_var="I_ext", output_var="s",
                        spike_var="spike", spike_def="v", op="qif_sfa_op",
                        node_vars={"k": 8.0, "tau": 2.0, "eta": -1.0}, v_reset=-1e2, v_peak=1e2, clear=False)

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
m = 3
p_in = 0.2

# generate input weights
W_in = input_connections(N, m, p_in, variance=5.0)

# add input layer
net.add_input_layer(m, weights=W_in, train=False)

# %%
# Here, we used the function :code:`rectipy.input_connections` to generate the coupling weights of the input layer.
# Again, the first two arguments control the dimensionality of the returned matrix, and the third argument controls the
# sparsity of the input weights. By default, the input weights are sampled from a standard Gaussian distribution,
# and :code:`variance=5.0` ensures that the variance of the Gaussian is set to :math:`\sigma^2 = 5` in this example.
# By toggling :code:`trainable`, it is possible to enable/disable optimization of the input weights during training.
#
# As a final step of the model initialization, lets add the readout layer:

# readout layer parameters
k = 3
activation_function = "softmax"

# add readout layer
net.add_edge(k, train=True, activation_function=activation_function)

# %%
# By declaring this layer as trainable, the weights of this layer are going to be optimized during training.
# We thus skipped on initializing the weights manually. Furthermore, we chose that the activation of the neurons in that
# layer are determined by the `softmax function <https://en.wikipedia.org/wiki/Softmax_function>`_,
# which is applied to the summed up synaptic inputs of each neuron in the layer.

# %%
# Step 2: Define inputs and targets
# ---------------------------------
#
# In the next step, we will define the extrinsic input that arrives at each unit of the input layer, as well as the
# target outputs that we would like our network to generate in response to this input. We will make the input a noisy
# signal, by generating spike trains via `Poisson processes <https://en.wikipedia.org/wiki/Poisson_point_process>`_
# with input rate parameters.

train_steps = 5000000
test_steps = 200000
steps = train_steps + test_steps
trial_steps = 10000
channels = list(np.arange(0, m, dtype=np.int32))
on_rate = 1.0
off_rate = 0.1
inp = np.zeros((steps, m))
active_channels = np.zeros_like(inp)
i = 0
while i*trial_steps < steps:
    in_channels = np.random.choice(channels, size=(2,), replace=False)
    active_channels[i*trial_steps:(i+1)*trial_steps, in_channels] = 1.0
    for c in range(m):
        if c in in_channels:
            inp[i*trial_steps:(i+1)*trial_steps, c] = np.random.poisson(lam=on_rate, size=(trial_steps,))
        else:
            inp[i * trial_steps:(i + 1) * trial_steps, c] = np.random.poisson(lam=off_rate, size=(trial_steps,))
    i += 1

# %%
# In the code above, we created Poisson spike trains for each channel (or input layer neuron). These spike trains
# had a higher average rate when drawn as an active input channel during a given trial. Let's have a look at the
# resulting input matrix:

import matplotlib.pyplot as plt

plt.imshow(inp[:100000].T, aspect=4000.0, interpolation="none")
plt.xlabel("training steps")
plt.ylabel("input channel")
plt.colorbar(label="# spikes", shrink=0.2)
plt.show()

# %%
# As can be seen, various combinations of input channels can be active at the same time (leading to higher spike rates
# in these channels). As target output data, we would like our network to recognize certain combinations of input channels.

targets = np.zeros((steps, k))
for i in range(steps):
    if active_channels[i, 0] * active_channels[i, 1] > 0:
        targets[i, 0] = 1.0
    elif active_channels[i, 0] * active_channels[i, 2] > 0:
        targets[i, 1] = 1.0
    elif active_channels[i, 1] * active_channels[i, 2] > 0:
        targets[i, 2] = 1.0

plt.imshow(targets[:100000].T, aspect=4000.0, interpolation="none")
plt.xlabel("training steps")
plt.ylabel("output channel")
plt.show()

# %%
# Step 3: Train the readout weights
# ---------------------------------
#
# Now, we have all pre-requisites to start our optimization procedure.
# To this end, we will use the `Network.train` method:

net.fit_bptt(inputs=inp[:train_steps], targets=targets[:train_steps], optimizer="rprop",
             loss="mse", lr=1e-2, update_steps=100000, record_output=False, record_loss=False,
             sampling_steps=steps, optimizer_kwargs={"etas": (0.5, 1.1), "step_sizes": (1e-4, 1e-1)})

# %%
# In this call to :code:`Network.train`, we chose to perform parameter optimization via the automated
# gradient calculation features provided by `torch`. We chose the `torch` version of the
# `resilient backpropagation algorithm <https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html>`_
# as an optimizer and the `mean-squared error <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_ as
# loss function. Furthermore, we chose an initial learning rate of :code:`lr=1e-2`, and provided some keyword arguments
# that control how the :code:`RProp` algorithm adjusts that learning rate. Finally, we turned off all variable
# recordings to speed up the training. Further performance improvements may be achieved by performing the training on
# the GPU, which can be done by providing the keyword argument :code:`device=`<device name>`.

# %%
# Step 4: Test the model performance
# ----------------------------------
#
# To test whether the training was successful, we can use the :code:`Network.test` method to examine how well the model
# can distinguish between different input combinations using a test data set:

samples = 500
obs, loss = net.test(inputs=inp[train_steps:], targets=targets[train_steps:], loss="mse",
                     record_output=True, record_loss=False, sampling_steps=samples, record_vars=[("s", False)])

print(f"Total loss on test data set: {loss}")

# %%
# As can be seen, the keyword arguments to :code:`Network.test` are very similar to the arguments of :code:`Network.train`.
# The main difference between the two calls is that we requested recordings of the network output and the RNN state
# variables :math:`s_i` every :code:`sampling_steps=500` test steps.
# The code below plots the network dynamics as well as a comparison between the network predictions and the targets
# on the test data:

_, ax = plt.subplots(figsize=(12, 6))
obs.matshow("s", interpolation="none", aspect=0.2, ax=ax)

fig, axes = plt.subplots(nrows=k, figsize=(12, 9))
predictions = np.asarray(obs["out"])
for i, ax in enumerate(axes):
    ax.plot(targets[train_steps::samples, i], "blue")
    ax.plot(predictions[:, i], "orange")
    plt.legend(["target", "prediction"])
ax.set_xlabel("time")
ax.set_ylabel("out")

plt.show()
