"""
Quadratic Integrate-and-Fire (QIF) Spiking Neuron Model
=======================================================

The QIF neuron model is a spiking point neuron model that simplifies the dynamics of a neuron to the evolution of
a single, dimensionless state variable :math:`v_i`:

.. math::

    \\tau \\dot v_i = v_i^2 + \\mu_i(t),

where :math:`\\tau` is a global time constant and :math:`\\mu_i` is a lumped representation of all inputs that arrive at
neuron :math:`i` at time :math:`t`. Any time :math:`v_i \\geq v_{peak}`, a spike is counted and the reset condition
:math:`v_i \\rightarrow v_{reset}` is applied. This introduces a discontinuity around the spike event to the dynamics
of :math:`v_i`, and makes the state variable similar to the membrane potential of a neuron close to the soma.
Remember, though, that :math:`v_i` is a dimensionless state-variable and thus can represent the membrane potential of a neuron
only up to an undefined scaling constant.
For a detailed description of the QIF neuron model, its dynamic regimes, and a derivation of this equation from more
complex models of neuronal dynamics, see [1]_ and [2]_.

Here, we will document the pre-implemented QIF neuron model available in `RectiPy` and how to use it as a base neuron
model for an RNN layer.
Currently, two different versions of the QIF model are implemented in `RectiPy`, both of which we will introduce below.

QIF neuron with synaptic dynamics
---------------------------------

The first version of the QIF neuron model available in `RectiPy` defines the input variable as follows

.. math::

    \\mu_i(t) &= \\eta_i + I_i(t) + \\tau s_i^{in}, \n
    \\tau_s \\dot s_i &= -\\frac{s}{\\tau_s} + \\delta(v_i - v_{peak}).

This definition of the input :math:`\\mu_i` allows for the usage of the QIF neuron as part of an RNN layer.
In the first equation, :math:`\\eta_i` represents a neuron-specific excitability, :math:`I_i` represents an extrinsic
input entering neuron :math:`i` at time :math:`t` (could be the input from a preceding layer), and :math:`s_i^{in}`
represents the combined input that this neuron receives from other QIF neurons.
The latter becomes more explicit in the second equation, which governs the dynamics of the synaptic output of the :math:`i^{th}` neuron.
Whereas the first term is a simple leakage term with synaptic integration time constant :math:`\\tau_s`, the second term
is 1 whenever the neuron spikes and 0 otherwise (:math:`\\delta` is the Dirac delta function).
Connections between different QIF neurons can be established by projecting :math:`s_i` to the variable :math:`s_i^{in}`
of other neurons in the network. We demonstrate this below.

References
^^^^^^^^^^

.. [1] E. Izhikevich (2007) *Dynamical Systems in Neuroscience.* MIT Press, ISBN: 978-0-262-09043-8.
.. [2] R. Gast (2021) *Phase transitions between asynchronous and synchronous neural dynamics.* University of Leipzig.
.. [3] R. Gast (2020) *A Mean-Field Description of Bursting Dynamics in Spiking Neural Networks with Short-Term Adaptation.* Neural Computation 32(9), 1615-1634.

"""

# %%
# Step 1: initialize a :code:`rectipy.Network` instance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We will start out by creating a random coupling matrix and implementing an RNN model of coupled QIF neurons.

from rectipy import Network
import numpy as np

# define network parameters
node = "neuron_model_templates.spiking_neurons.qif.qif"
N = 5
J = np.random.randn(N, N)*2.0

# initialize network
net = Network.from_yaml(node, weights=J, source_var="s", target_var="s_in", input_var="I_ext", output_var="s",
                        op="qif_op", dt=1e-3, spike_def="v", spike_var="spike")

# %%
# The above code initializes a network of :math:`N = 5` randomly coupled QIF neurons.
# We couple the variables :math:`s_i` to the variables :math:`s_i^{in}` via the coupling strengths given by :math:`J_{ij}`.
# Mathematically speaking, the above code implements :math:`s_i^{in} = \sum_{j=1}^N J_{ij} s_j`.
# In addition, we define the variable :math:`I_{ext}` as input variable of the RNN and :math:`s` as its output variable.

# %%
# Step 2: Simulate the RNN dynamics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, lets examine the network dynamics by integrating the evolution equations over 10000 steps, using the integration
# step-size of :code:`dt = 1e-3` as defined above.

# define network input
steps = 10000
inp = np.zeros((steps, N)) + 10.0

# perform numerical simulation
obs = net.run(inputs=inp, sampling_steps=10)

# %%
# We created a timeseries of constant input and fed that input to the input variable of the RNN at each integration step.
# Let's have a look at the resulting network dynamics.

from matplotlib.pyplot import show, legend

obs.plot("out")
legend([f"s_{i}" for i in range(N)])
show()

# %%
# As can be seen, the different QIF neurons generated different synaptic outputs even though they received the same
# extrinsic input, due to the random coupling and the RNN dynamics emerging from that.

# %%
# QIF neuron with spike-frequency adaptation
# ------------------------------------------
#
# The second QIF neuron model we provide with `RectiPy` incorporates a spike-frequency adaptation (SFA) mechanism.
# Specifically, the input :math:`\mu_i` is defined as
#
# .. math::
#
#       \mu_i(t) = \eta_i + I_i(t) - x_i + \tau s_i^{in},
#
# where :math:`x_i` represents a neuron-specific SFA variable, the dynamics of which are given by
#
# .. math::
#
#       \dot x_i = -\frac{x_i}{\tau_x} + \alpha \delta(v_i - v_{peak}),
#
# with adaptation time constant :math:`\tau_x` and adaptation strength :math:`\alpha`.
# The effects of SFA on the macroscopic dynamics of a QIF population are described in detail in [3]_.
# Here, we will show how they affect the RNN dynamics in a small QIF network.

# initialize network
node = "neuron_model_templates.spiking_neurons.qif.qif_sfa"
net = Network.from_yaml(node, weights=J, source_var="s", target_var="s_in", input_var="I_ext", output_var="s",
                        op="qif_sfa_op", dt=1e-3, spike_def="v", spike_var="spike")

# perform numerical simulation
obs = net.run(inputs=inp, sampling_steps=10)

# visualize the network dynamics
obs.plot("out")
legend([f"s_{i}" for i in range(N)])
show()

# %%
# As can be seen, the overall spiking activity in the network was clearly reduced by adding the SFA term
# (by default :math:`\alpha = 1.0` is used).