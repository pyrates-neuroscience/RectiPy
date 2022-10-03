"""
Leaky Integrator Rate Neuron Model
==================================

The `leaky integrator <https://en.wikipedia.org/wiki/Leaky_integrator>`_ (LI) is a standard model for neuronal dynamics.
It has been used for spiking as well as rate neuron models. Here, we introduce a rate-coupled LI neuron model.
It has a single state-variable :math:`v_i`, the dynamics of which are governed by

.. math::

    \\dot v_i &= -\\frac{v_i}{\\tau} + I_i(t) + k r_i^{in}, \n
    r_i &= f(v_i).

The two constants governing the dynamics of :math:`v_i` are the global decay time constant :math:`\\tau` and the global
coupling constant :math:`k`. The variable :math:`r_i` represents a potentially non-linear transform of the state-variable :math:`v_i`
and is used as a representation of the output rate of the neuron. This variable should be used to connect a neuron to other neurons
in an RNN. The variable :math:`r_i^{in}` serves as target variable for such connections.
`RectiPy` provides two different versions of the LI neuron that use two different choices for :math:`f`.
In the examples below, we examine an RNN of randomly coupled LI neurons for each of these two choices.

LI neuron with a hyperbolic tangent transform
---------------------------------------------

In this example, we will examine an RNN of LI neurons with :math:`r_i = \\tanh(v_i)`, i.e. the
`hyperbolic <https://en.wikipedia.org/wiki/Hyperbolic_functions>`_ tangent function
that represents a non-linear mapping of :math:`v_i` to the open interval :math:`(-1, 1)`.
"""

# %%
# Step 1: Network initialization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We will start out by creating a random coupling matrix and implementing an RNN model of rate-coupled LI neurons.

from rectipy import Network
import numpy as np

# define network parameters
node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
N = 5
J = np.random.randn(N, N)*1.5

# initialize network
net = Network.from_yaml(node, weights=J, source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                        output_var="li_op/v", dt=1e-3)

# %%
# The above code instantiates a `rectipy.Network` with :math:`N = 5` LI neurons with random coupling weights drawn from
# a standard Gaussian distribution with mean :math:`\mu = 0` and standard deviation :math:`\sigma = 2`. This particular
# choice of source and target variables for the coupling implements :math:`r_i^{in} = \sum_{j=1}^N J_{ij} r_j`.

# %%
# Step 2: Simulation of the network dynamics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, lets examine the network dynamics by integrating the evolution equations over 10000 steps, using the integration
# step-size of :code:`dt = 1e-3` as defined above.

# define network input
steps = 10000
inp = np.zeros((steps, N)) + 0.5

# perform numerical simulation
obs = net.run(inputs=inp, sampling_steps=10)

# %%
# We created a timeseries of constant input and fed that input to the input variable of the RNN at each integration step.
# Let's have a look at the resulting network dynamics.

from matplotlib.pyplot import show, legend

obs.plot("out")
legend([f"v_{i}" for i in range(N)])
show()

# %%
# As can be seen, the different LI neurons generated different output rates even though they received the same
# extrinsic input, due to the randomly sampled coupling weights and the RNN dynamics emerging from that.

# %%
# LI neuron with a logistic transform
# -----------------------------------
#
# In this example, we will examine an RNN of LI neurons with :math:`r_i = \frac{1}{1 + \exp(-v_i)}`, i.e. the
# `logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_ that represents a non-linear mapping of
# :math:`v_i` to the open interval :math:`(0, 1)`.

# initialize network
node = "neuron_model_templates.rate_neurons.leaky_integrator.sigmoid"
net = Network.from_yaml(node, weights=J, source_var="sigmoid_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                        output_var="li_op/v", dt=1e-3)

# perform numerical simulation
obs = net.run(inputs=inp, sampling_steps=10)

# visualize the network dynamics
obs.plot("out")
legend([f"v_{i}" for i in range(N)])
show()

# %%
# As can be seen, using the logistic function instead of the hyperbolic tangent function as an output function of the
# LI neurons led to a different network dynamics, even though both networks were initialized with the same coupling
# strengths and received the same extrinsic input.
