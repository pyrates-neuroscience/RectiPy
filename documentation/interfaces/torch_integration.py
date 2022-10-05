"""
Network-PyTorch Integration
===========================

While the methods `run`, `train`, and `test` of the `rectipy.Network` class provide wrappers to RNN simulation,
training, and testing via `torch`, the `Network` class and each of its layers can also be integrated into custom `torch`
code.

We will demonstrate this below for the `Network` class, using a simple optimization problem.

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
target.compile()

# test model initialization
learner = Network.from_yaml(node=node, weights=J, source_var="tanh_op/r", target_var="li_op/r_in",
                            input_var="li_op/I_ext", output_var="li_op/v", clear=True, dt=dt,
                            node_vars={"all/li_op/k": k_0, "all/li_op/tau": tau_0},
                            train_params=["li_op/k", "li_op/tau"])
learner.compile()

# prepare figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2)
ax[0].set_xlabel("training steps")
ax[0].set_ylabel("MSE")
ax[1].set_xlabel("training steps")
ax[1].set_ylabel(r"$v$")

# model fitting
import torch

opt = torch.optim.Rprop(learner.parameters(), lr=0.01, etas=(0.5, 1.1), step_sizes=(1e-5, 1e-1))
loss = torch.nn.MSELoss()
error, tol, step, update_steps, plot_steps = 1.0, 1e-4, 0, 1000, 100
mse_col, target_col, prediction_col = [], [], []
while error > tol:

    # calculate forwards
    I_ext = np.sin(np.pi * step * dt) * 0.5
    targ = target.forward(I_ext)
    pred = learner.forward(I_ext)
    step += 1

    # calculate loss and make optimization step
    l = loss(targ, pred)
    l.backward(retain_graph=True)
    if step % update_steps == 0:
        opt.step()
        opt.zero_grad()
    error = 0.9 * error + 0.1 * l.item()

    # plotting
    mse_col.append(error)
    target_col.append(targ.detach().numpy()[0])
    prediction_col.append(pred.detach().numpy()[0])
    if step % plot_steps == 0:
        ax[0].plot(mse_col, "red")
        ax[1].plot(target_col, "blue")
        ax[1].plot(prediction_col, "orange")
        fig.canvas.draw()
        fig.canvas.flush_events()
