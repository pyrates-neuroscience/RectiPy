"""
Network Observer
================

The `rectipy.observer.Observer` can be used to record neuron state variables during numerical simulations or
training/testing procedures performed with `rectipy.Network` instances. It is used by the methods `Network.run` and
`Network.test`, for example, which return a `rectipy.observer.Observer` instance with state variable recordings.

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

import pandas as pd
import numpy as np
from rectipy import Network

# network parameters
N = 5
J = np.random.rand(N, N) * 20.0
node = "neuron_model_templates.spiking_neurons.qif.qif"

# network initialization
net = Network(dt=1e-3)
net.add_diffeq_node("qif", node, weights=J, source_var="s", target_var="s_in", input_var="I_ext", output_var="s",
                    op="qif_op", spike_def="v", spike_var="spike")

# %%
# Step 2: Observer initialization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next, we need to initialize our observer. This can be done as follows:

from rectipy import Observer

obs = Observer(net.dt, record_output=True, record_loss=False, record_vars=[("qif", "v", False)])

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
# a separate variable to record. List entries should be passed as tuples, where the first tuple entry is the name of
# the network node, the second tuple entry is the name of the variable on that node, and the third entry indicates
# whether to record this variable for every neuron in the RNN layer, or whether to record the average of that variable
# across all neurons. Choosing the latter option leads to reduced memory consumption.

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
    obs.record(step, output=out, loss=0.0, record_vars=[net.get_var("qif", "v")])

# %%
# In that example, we recorded the network output variable :math:`s` as well as the QIF neuron state variable :math:`v`
# at each integration step. For the latter, we made use of the `RNNLayer.record` method which yields the current state
# of each of the variables passed in the list.
# Note that the names of the variables that have been declared as :code:`record_vars` during the initialization of the
# Observer are also available via the attribute :code:`obs.recorded_rnn_variables`. Thus, you could also provide the
# keyword argument :code:`record_vars=[net[v] for v in obs.recorded_rnn_variables]` to the :code:`Observer.record` method.
#
# After this procedure, we can visualize our recordings via the `Observer.plot` method, which allows you to either plot
# state variables against time or against each other.

import matplotlib.pyplot as plt

obs.plot(("qif", "v"))
plt.show()

obs.plot(y=("qif", "v"), x="out")
plt.show()

# %%
# Note that the network output is always available via "out", whereas additional recorded variables are available via
# a tuple of the name of the network node ("qif) and the name of the variable on that node ("v").
# Alternatively, you can simply retrieve the recordings from the observer for subsequent analysis/plotting via your own
# custom scripts. To this end, you have several options:

v = obs.to_dataframe(("qif", "v"))  # type: pd.DataFrame
s_raw = obs["out"]  # type: list
s_numpy = obs.to_numpy("out")  # type: np.ndarray

# %%
#
# The `Observer.to_dataframe` method returns a `pandas.DataFrame` with the recorded variable, the indexing method
# returns a list with the recorded `torch.Tensor` objects, and `Observer.to_numpy` returns a numpy array with the
# stacked results.
#
# Using the Observer in custom scripts
# ------------------------------------
#
# Standard numerical simulations, model training, and model testing methods are provided by `rectipy.Network`
# via its methods `run`, `train`, and `test`. You will not have to bother with observer initialization and
# manual variable recordings if you are using these methods. Instead, you can control the behavior of the observer
# via keyword arguments to these methods. The keyword arguments are the same for each of these methods and we will
# demonstrate how to use them via the `Network.run` method. The code below performs the same simulation that we
# performed manually above.

obs = net.run(inputs=inp, record_vars=[("qif", "v", True)], verbose=False)

# %%
# As additional options, you can change the sampling step-size of your recordings:

obs2 = net.run(inputs=inp, record_vars=[("qif", "v", True)], sampling_steps=2, verbose=False)
print(len(obs[("qif", "v")]))
print(len(obs2[("qif", "v")]))
ax = obs.plot(("qif", "v"))
obs2.plot(("qif", "v"), ax=ax)
plt.legend(["obs", "obs2"])
plt.show()

# %%
# As you can see, the second observer stored the state variable :math:`v` at only every second integration step, when
# :code:`sampling_steps=2` was given.
# You can also toggle storage of the output variable and loss on and off, using the same keyword arguments as for the
# observer initialization:

net.run(inputs=inp, record_output=True, record_loss=False, verbose=False)
