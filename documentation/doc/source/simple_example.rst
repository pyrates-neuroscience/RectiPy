**************************************************
Example: Simulation of an RNN with neuron dynamics
**************************************************

In this short example, we will perform a numerical simulation of a network of leaky integrator neurons with rate-based, recurrent coupling:

.. math::
        \dot u_i = -\frac{u_i}{\tau} + I_i(t) + k \sum_{j=1}^N J_{ij} \tanh(u_j)


where :math:`u_i` is the state variable of the :math:`i^{th}` neuron in a network of :math:`N` neurons.
Without any input, the activity of each neuron decays to zero with global decay time constant :math:`\tau`.
Input enters the neuron either trough the extrinsic input :math:`I_i` (could be the input from a preceding neuron layer, for example),
or though the other neurons in the network. The latter is realized by the sum over all network units.
The coupling is non-linear, due to the hyperbolic tangent transform, and scales with the global coupling constant :math:`k` and
the connection-specific coupling constants :math:`J_{ij}`.

Similar network formulations have been used as a simplified model for recurrent neural networks in the brain.
Since this model comes pre-implemented as a template in `RectiPy`, we won't have to implement the model equations.
Instead, the following code suffices to implement a model of :math:`N=100` randomly coupled leaky-integrator neurons:

.. code-block::

    from rectipy import Network
    import numpy as np

    # model parameters
    model = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    N = 100
    J = np.random.randn(N, N)
    dt = 1e-3

    # initialize Network instace
    net = Network(dt=dt, device="cpu")

    # add a population of N coupled rate neurons to the network
    net.add_diffeq_node("tanh", node, weights=J, source_var="tanh_op/r",
                        target_var="li_op/r_in", input_var="li_op/I_ext",
                        output_var="li_op/v")


We simply had to pass the pointer to the model template and the weight matrix, and provide the names of the variables that
we wanted to couple recurrently via the weights (:code:`source_var` to :code:`target_var`) and the names of the variables
that we wanted to declare as the input and output of the rate neuron population.
The specific names of these variables depend on the structure of the model template, which is explained in detail in the
`PyRates documentation <https://pyrates.readthedocs.io/en/latest/template_specification.html>`_.
For a more detailed explanation of how to use the :code:`rectipy.Network.add_diffeq_node` method, see this
`use example <https://rectipy.readthedocs.io/en/latest/auto_interfaces/model_definition.html>`_..

In a next step, we could either add additional input and output layers to the :code:`Network` instance, use the :code:`Network.fit_bptt`
method to fit some of its parameters via backpropagation throug time, or perform numerical simulations via :code:`Network.run`.
Here, we will do the latter. Concretely, we will use the standard `Euler method <https://en.wikipedia.org/wiki/Euler_method>`_ to numerically solve the
`initial value problem <http://www.scholarpedia.org/article/Initial_value_problems>`_ :math:`u_i(t) = \int_{t0}^t \dot u_i dt'`,
given initial time point :math:`t_0` and initial network state :math:`u_i(t_0) \forall i`.
This can be done as follows:

.. code-block::

    # define input
    steps = 10000
    inp = np.random.ones((steps, N))

    # perform numerical simulation
    obs = net.run(inputs=inp)


Here, we performed the numerical simulation and unit input at each integration step for simplicity.
The :code:`Network.run` method returns a `rectipy.observer.Observer` instance that contains
all the simulated time series and can be used for subsequent plotting of the integration results.
To plot the output variable of the RNN layer, simple call

.. code-block::

    obs.plot("out")
