"""
Leaky Integrate-and-Fire Spiking Neuron Model
=============================================

The `leaky integrate-and-fire <https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire>`_ (LIF)
neuron model is the most commonly used spiking neuron model.
It has a single state-variable :math:`v_i`, the dynamics of which are governed by

.. math::

    \\dot v_i &= -\\frac{v_i}{\\tau} + I_i(t) + k s_i^{in}.

The two constants governing the dynamics of :math:`v_i` are the global decay time constant :math:`\\tau` and the global
coupling constant :math:`k`. Any time :math:`v_i \\geq v_{peak}`, a spike is counted and the reset condition
:math:`v_i \\rightarrow v_{reset}` is applied. This introduces a discontinuity around the spike event to the dynamics
of :math:`v_i`, and makes the state variable similar to the membrane potential of a neuron close to the soma.
Spikes or spike-driven synapses should be used as input to other neurons in an RNN.
Here, we use a spike-driven synaptic activation :math:`s_i` with the following dynamics:

.. math::

    \\tau_s \\dot s_i &= -\\frac{s_i}{\\tau_s} + \\delta(v_i - v_{peak}),

where :math:`\\tau_s` is the synaptic time constant and :math:`\\delta` is the `Dirac delta <https://en.wikipedia.org/wiki/Dirac_delta_function>`_ function,
which is one whenever a spike is elicited by the :math:`i^{th}` neuron and zero otherwise.
The variable :math:`s_i^{in}` serves as target variable for such synaptic activation variables.

In this example, we will examine an RNN of LIF neurons with spike-coupling.
"""

# %%
# Step 1: Network initialization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We will start out by creating a random coupling matrix and implementing an RNN model of rate-coupled LIF neurons.

from rectipy import Network
import numpy as np

# define network parameters
node = "neuron_model_templates.spiking_neurons.lif.lif"
N = 5
J = np.random.randn(N, N)*20.0

# initialize network
net = Network.from_yaml(node, weights=J, source_var="s", target_var="s_in", input_var="I_ext",
                        output_var="s", spike_var="spike", spike_def="v", op="lif_op")

# %%
# The above code instantiates a `rectipy.Network` with :math:`N = 5` LIF neurons with random coupling weights drawn from
# a standard Gaussian distribution with mean :math:`\mu = 0` and standard deviation :math:`\sigma = 10`. This particular
# choice of source and target variables for the coupling implements :math:`s_i^{in} = \sum_{j=1}^N J_{ij} s_j`.

# %%
# Step 2: Simulation of the network dynamics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, lets examine the network dynamics by integrating the evolution equations over 10000 steps, using the integration
# step-size of :code:`dt = 1e-3` as defined above.

# define network input
steps = 10000
inp = np.zeros((steps, N)) + 200.0

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
# As can be seen, the different LIF neurons spiked at different times, even though they received the same
# extrinsic input, due to the randomly sampled coupling weights and the RNN dynamics emerging from that.
