"""
Network Simulations
===================

`RectiPy` provides basic support for numerical simulations of network dynamics.
Neuron models of the RNN layer are governed by differential equation systems of the form

.. math::

    \\dot y = F(y, \\theta, t)

With state-vector :math:`y`, parameter vector :math`\\theta`, vector-field :math:`F` and time :math`t`.
Thus, when we speak of numerical simulations of the network dynamics, we refer to solving the
`initial value problem <https://en.wikipedia.org/wiki/Initial_value_problem>`_ (IVP) via numerical methods.
Given an initial time :math:`t_0` and an initial state :math:`y(t_0)`, the IVP amounts to finding the solution

.. math::

    y(t) = \\int_{t' = t_0}^{t} F(y, \\theta, t') dt'

To solve this integral numerically, we apply the standard `Euler method <https://en.wikipedia.org/wiki/Euler_method>`_.
This procedure is available to the user via the method `rectipy.Network.run`. We will show how to apply this method,
using the example of an RNN of QIF neurons. Have a look at our `documentation of the QIF neuron
<https://rectipy.readthedocs.io/en/latest/auto_models/qif.html>`_ for details on its
mathematical definition and implementation in `RectiPy`.

Step 1: Network initialization
------------------------------

As a first step, let's define a network of :math:`N = 5` randomly coupled QIF neurons.
"""

from rectipy import Network
import numpy as np

# network initialization
N = 5
J = np.random.rand(N, N) * 20.0
dt = 1e-3
qif = Network.from_yaml("neuron_model_templates.spiking_neurons.qif.qif", weights=J, dt=dt,
                        source_var="s", target_var="s_in", input_var="I_ext", output_var="s",
                        spike_var="spike", spike_def="v", op="qif_op")

# %%
# An important variable for numerical integration is the integration step-size :code:`dt`. It's default value is
# :code:`dt=1e-3`, but depending on the smallest time constant in your model, you might have to choose a smaller value.
# Its physical unit depends on the unit of the time constants in your model. In this example, :code:`dt` is defined in
# units of :math:`\tau`, the evolution time constant of the QIF neuron.

# %%
# Step 2: Define extrinsic inputs
# -------------------------------------
#
# We would like to solve the IVP using the `Network.run` method over a time interval of :math:`T = 10`
# (again, in units of :math:`\tau`). Due to :code:`dt = 1e-3`, this amounts to 10000 integration steps.
# `RectiPy` requires us to define the extrinsic input to the network for each integration step.
# Here, we apply a sine wave input to some of the target neurons, for demonstration purposes.

# initialize input array
steps = 10000
time = np.arange(0, steps) * dt
inp = np.zeros((steps, N))

# define target neurons, input strengths, and input frequencies
target_neurons = [2, 4]
inp_strengths = [20.0, 10.0]
inp_freqs = [0.5, 0.25]

# add sine wave inputs to input array
for n, amp, freq in zip(target_neurons, inp_strengths, inp_freqs):
    inp[:, n] = np.sin(2*np.pi*freq*time)*amp

# plot the inputs
import matplotlib.pyplot as plt
plt.plot(inp)
plt.legend([f"inp_{i}" for i in target_neurons])
plt.show()

# %%
# Step 3: Simulate the network dynamics
# -------------------------------------
#
# Now that we have defined the input, lets solve the IVP using the `rectipy.Network.run` method:

obs = qif.run(inputs=inp)

# %%
# The return value of that method is a `rectipy.observer.Observer` instance, which is described in detail
# `in another use example <https://rectipy.readthedocs.io/en/latest/auto_interfaces/observer.html>`_.
# Importantly, it records the output variable of the RNN at each integration step by
# default. We can visualize these dynamics as follows:

obs.plot("out")
plt.legend([f"s_{i}" for i in range(N)])
plt.show()

# %%
# As can be seen, the extrinsic input pushed the QIF network into a high-activity regime of relatively
# synchronized spiking activity.
# Next to the :code:`inputs` argument, the `rectipy.Network.run` method provides additional keyword arguments
# that control the recording of RNN state variables. These are described in more detail in the
# use example that covers the `rectipy.observer.Observer <https://rectipy.readthedocs.io/en/latest/auto_interfaces/observer.html>`_ class.
# Finally, the `rectipy.Network.run` method allows to toggle the progress display via the
# :code:`verbose` keyword argument, and allows to choose the device you would like the numerical integration to be performed on
# via the :code:`device` keyword argument:

obs = qif.run(inp, verbose=False, device="cpu")
