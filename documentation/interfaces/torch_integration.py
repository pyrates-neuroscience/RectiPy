"""
Network-PyTorch Integration
===========================

While the methods `run`, `train`, and `test` of the `rectipy.Network` class provide wrappers to RNN simulation,
training, and testing via `torch`, the `Network` class and each of its layers can also be integrated into custom `torch`
code.

We will demonstrate this below for the `Network` class, using a simple optimization problem.
Specifically, we will perform online parameter optimization in a model of rate-coupled leaky integrator neurons of the
form

.. math::

    \\dot v_i &= -\\frac{v_i}{\\tau} + I_i(t) + k r_i^{in}, \n
    r_i &= tanh(v_i).

This rate neuron model is described in detail in `this use example <https://rectipy.readthedocs.io/en/latest/auto_models/leaky_integrator.html>`_.
For our optimization problem, we will focus on the global leakage time constant :math:`\\tau` and the global coupling
constant :math:`k`.
We will set up two separate :code:`rectipy.Network` instances, a target network with the target values of :math:`\\tau`
and :math:`k`, and a learner network with a different, randomly sampled set of values for :math:`\\tau` and :math:`k`.
We will then simulate the dynamic response of both network to a periodic extrinsic driving signal and optimize :math:`\\tau`
and :math:`k` of the learner network such that its dynamics resembles the dynamics of the target network.

Step 1: Network initialization
------------------------------

First, lets set up both networks with different parametrizations for :math:`\\tau` and :math:`k`.
"""

import numpy as np
from rectipy import Network

# network parameters
node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
N = 5
dt = 1e-3
J = np.random.randn(N, N)
k_t = np.random.uniform(0.25, 4.0)
tau_t = np.random.uniform(0.25, 4.0)
k_0 = np.random.uniform(0.25, 4.0)
tau_0 = np.random.uniform(0.25, 4.0)

# target model initialization
target = Network.from_yaml(node=node, weights=J, source_var="tanh_op/r", target_var="li_op/r_in",
                           input_var="li_op/I_ext", output_var="li_op/v", clear=True, dt=dt,
                           node_vars={"all/li_op/k": k_t, "all/li_op/tau": tau_t})

# test model initialization
learner = Network.from_yaml(node=node, weights=J, source_var="tanh_op/r", target_var="li_op/r_in",
                            input_var="li_op/I_ext", output_var="li_op/v", clear=True, dt=dt,
                            node_vars={"all/li_op/k": k_0, "all/li_op/tau": tau_0},
                            train_params=["li_op/k", "li_op/tau"])

print("Target network parameters: " + r"$k_t$ = " + f"{k_t}" + r", $\tau_t$ = " + f"{tau_t}.")
print("Learner network parameters: " + r"$k_0$ = " + f"{k_0}" + r", $\tau_0$ = " + f"{tau_0}.")

# %%
# As can be seen, we drew two different sets of parameters for our networks.

# %%
# Step 2: Perform online optimization
# -----------------------------------
#
# Now, we would like to optimize the parameters of our learner network in an online optimization algorithm.
# We will do this via a custom `torch` optimization procedure. As a first step, we need to compile both networks
# to be able to use them as `torch` modules:

target.compile()
learner.compile()

# %%
# Next, we are going to choose an optimization algorithm:

import torch

opt = torch.optim.Rprop(learner.parameters(), lr=0.01, etas=(0.5, 1.1), step_sizes=(1e-5, 1e-1))

# %%
# We chose the `resilient backpropagation algorithm <https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html>`_
# and tweaked some of its default parameters to control the automated learning rate adjustments.
# In addition, we need to specify a loss function:

loss = torch.nn.MSELoss()

# %%
# Here, we just chose the vanilla `mean-squared error <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_.
# Finally, lets initialize a figure in which we are going to plot the progress of the online optimization:

# matplotlib settings
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.ion()

# figure layout
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
ax[0].set_xlabel("training steps")
ax[0].set_ylabel("MSE")
ax[1].set_xlabel("training steps")
ax[1].set_ylabel(r"$v$")

# %%
# Now, we are ready to perform the optimization. The code below implements a torch optimization procedure that adjusts
# the parameters :math:`\tau` and :math:`k` of the learner network every :code:`update_steps` steps, based on the
# automatically backpropagated mean-squared error between the outputs of the target and learner network in response to a
# sinusoidal extrinsic input. The optimization will run until convergence, or until a maximum number of optimization
# steps has been reached.

# model fitting
error, tol, step, update_steps, plot_steps, max_step = 10.0, 1e-5, 0, 1000, 100, 1000000
mse_col, target_col, prediction_col = [], [], []
while error > tol and step < max_step:

    # calculate network outputs
    I_ext = np.sin(np.pi * step * dt) * 0.5
    targ = target.forward(I_ext)
    pred = learner.forward(I_ext)
    step += 1

    # calculate loss
    l = loss(targ, pred)
    l.backward(retain_graph=True)

    # make optimization step
    if step % update_steps == 0:
        opt.step()
        opt.zero_grad()

    # update average error
    error = 0.95 * error + 0.05 * l.item()

    # collect data
    mse_col.append(error)
    target_col.append(targ.detach().numpy()[0])
    prediction_col.append(pred.detach().numpy()[0])

    # update the figure for online plotting
    if step % plot_steps == 0:
        ax[0].plot(mse_col, "red")
        ax[1].plot(target_col, "blue")
        ax[1].plot(prediction_col, "orange")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

# retrieve optimized parameters
params = list(learner.parameters())
k = params[0].clone().detach().numpy()
tau = params[1].clone().detach().numpy()
print("Optimized parameters: " + r"$k_*$ = " + f"{k[0]}" + r", $\tau_*$ = " + f"{tau[0]}.")

# %%
# The code above demonstrates how any :code:`rectipy.Network` instance can be integrated into custom `torch` code.
# After calling :code:`Network.compile`, the :code:`Network` instance provides the standard :code:`torch.nn.Module.forward` and
# :code:`torch.nn.Module.parameters` methods that you can use to calculate the network output and access the trainable
# parameters, respectively. The same holds for each layer of the :code:`Network`: :code:`rectipy.input_layer.InputLayer`,
# :code:`rectipy.output_layer.OutputLayer`, and :code:`rectipy.rnn_layer.RNNLayer`. This allows to implement more complex
# optimization procedures that go beyond the functions that :Code:`Network.train` provides.
