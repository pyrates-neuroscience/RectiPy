"""
Network Initialization
======================

`RectiPy` models are generally initialized via the :code:`rectipy.Network` class. The method `Network.from_yaml` serves
as the main user interface for network model initialization.
It allows the user to define the dynamic equations and default parameters of the units of an RNN via `YAML` templates.
To this end, `RectiPy` makes use of the model definition capacities of `PyRates <https://github.com/pyrates-neuroscience/PyRates>`_.
`PyRates` provides a detailed documentation that covers `the mathematical syntax <https://pyrates.readthedocs.io/en/latest/math_syntax.html>`_
that can be used to define neuron models, and how to use their `YAML template classes <https://pyrates.readthedocs.io/en/latest/template_specification.html>`_
to define neurodynamic models. We refer readers to this documentation to learn how to define neuron model templates themselves.

Here, we will make use of the pre-implemented neuron models that come with `RectiPy` to explain how to initialize a
`rectipy.Network` based on a neuron model template and how the `rectipy.Network` can be customized afterwards to
construct a neural network model of choice.

How to create a network with differential-equation-based nodes
--------------------------------------------------------------

Let's begin with the `rectipy.Network.add_diffeq_node` method.
The code below initializes a `rectipy.Network` instance and adds :math:`N = 5` rate-coupled LI neurons to the network via this method.
LI rate neurons are defined via a single ordinary differential equation.
For an introduction to the LI rate neuron model, see `the respective use example <https://rectipy.readthedocs.io/en/latest/auto_models/leaky_integrator.html>`_.
"""

from rectipy import Network
import numpy as np

# define network parameters
node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
N = 5
J = np.random.randn(N, N)*2.0
dt = 1e-3

# initialize network
net = Network(dt=dt, device="cpu")

# add a rate-neuron population to the network
net.add_diffeq_node("tanh", node, weights=J, source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                    output_var="li_op/v")

# %%
# In the code above, the variable :code:`node` is a path pointing to a YAML template that defines the dynamic equations and
# variables of a single LI neuron. Since `RectiPy` comes with a module that contains all its pre-implemented models
# (called `neuron_model_templates`), we can use the :code:`.` syntax to specify that path. If you define your own
# neuron model, you can use standard Python syntax to specify the path to that model, i.e. :code:`node = "<path>/<yaml_file_name>.<template_name>"`.
# As a second important piece of our network definition, we need to provide an :math:`N x N` coupling matrix, that defines
# the structure of the recurrent coupling in a network of :math:`N` neurons. Above, we call this matrix :code:`J`.
# During the call to `rectipy.Network.add_diffeq_node`, a `rectipy.nodes.RateNet` instance will be created, which can be accessed via
# `net["tanh"]` and contains a network of :math:`N` recurrently coupled neurons,
# with the neuron type given by :code:`node` and the coupling weights given by :code:`J`.
# The variables :code:`source_var` and :code:`target_var` specify which variables of the neuron model to use to establish the
# recurrent coupling. In the above example, we specify to project the variable :code:`r` of each neuron,
# defined in operator template :code:`tanh_op`, to the variable :code:`r_in` of each neuron, defined in operator template :code:`li_op`.
# The coupling weights provided via the keyword argument :code:`weights` will be used to scale the :code:`source_var`, and multiple
# sources projecting to the same target will be summed up. Thus, the above code implements the following definition of
# the target variable :code:`r_in` of each neuron: :math:`r_i^{in} = \sum_{j=1}^N J_{ij} r_j`.
#
# Furthermore, we specified that the variable :code:`I_ext` defined in operator template :code:`li_op` should serve as an
# input variable for any extrinsic inputs that we would like to apply to the RNN. We could use this variable, for example,
# to provide input from another network layer to the RNN. Finally, we defined that the variable :code:`u` defined in
# operator template :code:`li_op` serves as the output of the RNN layer and can thus be used to connect it to subsequent
# network layers.
#
# Below, we show how to access a couple of relevant attributes of the network. For a full documentation of the
# different attributes of `rectipy.network.Network` and `rectipy.nodes.RateNet` instances, see our `API <https://rectipy.readthedocs.io/en/latest/rectipy.html>`_.

net.compile()

print(f"Number of neurons in network: {net.n_out}")
print(f"Network nodes: {net.nodes}")
print(f"State variables of the network: {net.state}")
print(f"LI population parameters: {net['tanh']['node'].parameter_names}")

# %%
# How to create a spiking neural network
# --------------------------------------
#
# in the above example, we initialized a network of rate neurons. To initialize a network of spiking neurons instead,
# we need to provide two additional keyword arguments to `Network.add_diffeq_node`:

# change neuron model to leaky integrate-and-fire neuron
node = "neuron_model_templates.spiking_neurons.lif.lif"

# initialize network
net = Network(dt=dt, device="cpu")

# initialize network
net.add_diffeq_node("lif", node, weights=J, source_var="s", target_var="s_in", input_var="I_ext",
                    output_var="s", op="lif_op", spike_var="spike", spike_def="v", spike_threshold=100.0,
                    spike_reset=-100.0)

# %%
# First, note that we did not provide the operator template names with the variable names in this call to `Network.add_diffeq_node`.
# Instead, we provided the operator name as a separate keyword argument :code:`op`. This is possible in this case,
# since the LIF neuron template only consists of a single operator template.
#
# The keyword argument :code:`spike_var` specifies which variable to use to store spikes in, whereas the keyword
# argument :code:`spike_def` specifies which state variable to use to define the spiking condition of a neuron.
# When these two keyword arguments are provided, `rectipy.Network` is initialized with a `rectipy.nodes.SpikeNet`
# instance instead of a `rectipy.nodes.RateNet` instance. The class `rectipy.nodes.SpikeNet` implements the
# following spike condition:
#
# .. code-block::
#       if spike_def > spike_threshold:
#           spike_var = 1.0
#           spike_def = spike_reset
#       else:
#           spike_var = 0.0
#
# In more mathematic terms, and specific to our example model, this means that a spike is counted when :math:`v_i \geq \theta`,
# where :math:`\theta` is the spiking threshold. A spike is then counted via the variable :code:`spike_var` and the neuron's
# state variable is reset: :math:`v_i \rightarrow v_r`.
# The two control parameters of this spike reset conditions, :math:`\theta` and :math:`v_r`,  can be controlled via two
# additional keyword arguments: :code:`spike_threshold` and :code:`spike_reset`.
# The default parameters of these two keyword arguments are as specified in the above call to `Network.add_diffeq_node`.
# Let's inspect the differences between the spiking network RNN layer and the rate-neuron RNN layer.

net.compile()

print(f"Number of neurons in network: {net.n_out}")
print(f"Network nodes: {net.nodes}")
print(f"State variables of the network: {net.state}")
print(f"LI population parameters: {net['lif']['node'].parameter_names}")

# %%
# As can be seen, a `SpikeNet` instance is now the only network node instead of a `RateNet` instance.
# Furthermore, the network node has a different state vector and a different number of parameters, owing to the differences
# in the LIF neuron as compared to the LI rate neuron. The state variable difference comes from the additional
# state-variable :math:`s_i`, whereas the difference in the number of model parameters is caused by the synaptic time
# constant :math:`\tau_s`.

# %%
# How to add a simple function node to the network
# ------------------------------------------------
#
# In addition to nodes that are based on differential equation systems, it is also possible to add nodes with instantaneous
# activation functions. The code below shows how to add a node with the identity function (can be useful as input layers):

m = 3
net.add_func_node("input", m, activation_function="identity")

# %%
# To connect this node to an existing node, we can use the `Network.add_edge` method:

net.add_edge(source="input", target="lif")

# %%
# In this most simple case, a random set of weights will be drawn to map from the :math:`m = 3` input nodes to the :math:`N = 5` LIF neurons.
# You can also specify weights yourself and pass them via the keyword argument :code:`weights`.
# Let's add another function node and specify the weights explicitly:

# add a node of sigmoid units
k = 2
net.add_func_node("output", k, activation_function="sigmoid")

# add an edge from the LIF population to the sigmoid node
net.add_edge(source="lif", target="output", weights=np.random.randn(N, k))
