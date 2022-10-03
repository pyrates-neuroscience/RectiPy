"""
Network Observer
================

The `rectipy.observer.Observer` can be used to record neuron state variables during numerical simulations or
training/testing procedures performed with `rectipy.Network` instances. It is used by the methods `Network.run`,
`Network.train`, and `Network.test`, which all return a `rectipy.observer.Observer` instance with state variable
recordings.

Using the Observer in custom scripts
------------------------------------

Here, we will demonstrate how an `Observer` instance can be used within a custom simulation of the dynamics of a network
of :math:`N=5` randomly coupled QIF neurons.
For a detailed introduction to the neuron model used in this example, see our documentation of the
`QIF spiking neuron <https://rectipy.readthedocs.io/en/latest/auto_models/qif.html>`_.

Step 1: Network initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let's set up the network.
"""

import numpy as np
from rectipy import Network

# network parameters
N = 5
J = np.random.rand(N, N) * 20.0
node = "neuron_model_templates.spiking_neurons.qif.qif"

# network initialization
net = Network.from_yaml(node, weights=J, source_var="s", target_var="s_in", input_var="I_ext", output_var="s",
                        op="qif_op", spike_def="v", spike_var="spike",
                        record_vars=["v"])

# %%
# Note that we provided the keyword argument :code:`record_vars` in our call to `Network.from_yaml`. You can use this
# keyword argument to declare any additional state variables of the model that you would like to record apart from the
# variable declared as :code:`output_var`. The latter will be recorded automatically during calls to `Network.run`,
# if not manually turned off.

# %%
# Step 2: Observer initialization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next, we need to initialize our observer. This can be done as follows:

from rectipy import Observer

obs = Observer(net.rnn_layer.dt, record_output=True, record_loss=False, record_vars=[("v", False)])

# %%
# The first argument to `Observer` is the integration step-size that we will use to solve the differential equations of
# our network. In addition, we told the observer to record the output variable of the network via
# :code:`record_output=True`. In our example, this is the variable :code:`s` of the neurons in our RNN layer. However,
# if we were to add an output layer to the network (see `this <https://rectipy.readthedocs.io/en/latest/auto_interfaces/model_definition.html>`_
# example for information on how to do that), the output of that output layer would be recorded instead.
# We will demonstrate this at a later stage. The :code:`record_loss` keyword argument can be toggled on, if you
# would like to record the loss during a training or testing procedure (see `this <https://rectipy.readthedocs.io/en/latest/auto_interfaces/train_test.html>`_
# example for details on model training). Finally, the keyword argument :code:`record_vars` serves to declare which
# additional state variables the observer should stage for recording. It is passed as a list, where each list entry is
# a separate variable to record. List entries should be passed as tuples, where the first tuple entry is the name of the variable,
# whereas the second entry indicates whether to record this variable for every neuron in the RNN layer, or whether to
# record the average of that variable across all neurons. Choosing the latter option leads to reduced memory consumption.

# %%
# Step 3: Perform recordings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that we have the network and observer initialized, we can use both of them to simulate the network dynamics and
# record the evolution of its state variables.

import torch

# define input
steps = 10000
inp = torch.zeros((steps, N)) + 15.0

# compile the network
net.compile()

# perform numerical simulation
for step in range(steps):
    out = net.forward(inp[step, :])
    obs.record(step, output=out, loss=0.0, record_vars=[net["v"]])

# %%
# In that example, we recorded the network output variable :math:`s` as well as the QIF neuron state variable :math:`v`
# at each integration step. For the latter, we made use of the `RNNLayer.record` method which yields the current state
# of each of the variables passed in the list.
#
# After this procedure, we can visualize our recordings via the `Observer.plot` method, which allows you to either plot
# state variables against time or against each other.

import matplotlib.pyplot as plt

obs.plot("v")
plt.show()

obs.plot(y="v", x="out")
plt.show()

# %%
# Alternatively, you can simply retrieve the recordings from the observer for subsequent analysis/plotting via your own
# custom scripts:

v = obs["v"]
s = obs["out"]

# %%
# Using the Observer in custom scripts
# ------------------------------------
#
# Standard numerical simulations, model training, and model testing methods are provided by `rectipy.Network`
# via its methods `run`, `train`, and `test`. You will not have to bother with observer initialization and
# manual variable recordings if you are using these methods. Instead, you can control the behavior of the observer
# via keyword arguments to these methods. The keyword arguments are the same for each of these methods and we will
# demonstrate how to use them via the `Network.run` method. The code below performs the same simulation that we performed
# manually above.

obs = net.run(inputs=inp, record_vars=[("v", True)], verbose=False)

# %%
# As additional options, you can change the sampling step-size of your recordings:

obs2 = net.run(inputs=inp, record_vars=[("v", True)], sampling_steps=2, verbose=False)
print(len(obs["v"]))
print(len(obs2["v"]))
ax = obs.plot("v")
obs2.plot("v", ax=ax)
plt.legend(["obs", "obs2"])
plt.show()

# %%
# As you can see, the second observer stored the state variable :math:`v` at only every second integration step, when
# :code:`sampling_steps=2` was given.
# You can also toggle storage of the output variable and loss on and off, using the same keyword arguments as for the
# observer initialization:

net.run(inputs=inp, record_output=True, record_loss=False, verbose=False)
