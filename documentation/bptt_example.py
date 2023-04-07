import matplotlib.pyplot as plt

from rectipy import Network
import numpy as np
from matplotlib.pyplot import show


# model parameters
node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
N = 200
k = 2.0
tau = np.random.uniform(10.0, 20.0, size=(N,))
eta = 2.0
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
dt = 1e-2
node_vars = {"all/li_op/eta": eta, "all/li_op/tau": tau, "all/li_op/k": k}

# initialize target network
target_net = Network(dt, device="cpu")
target_net.add_diffeq_node_from_yaml("tanh", node=node, weights=J0, source_var="tanh_op/r",
                                     target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v",
                                     clear=True, float_precision="float64", node_vars=node_vars)

# simulate target time series
T = 100.0
steps = int(T/dt)
inp = np.zeros((steps, 1))
time = np.linspace(0, T, steps)
inp[:, 0] = np.sin(2.0*np.pi*0.2*time) * 10.0
target_obs = target_net.run(inp, sampling_steps=1)
target = target_obs.to_numpy("out")

# initialize learner net
J1 = np.random.randn(N, N)
J1 /= np.max(np.abs(np.linalg.eigvals(J1)))
learner_net = Network(dt, device="cpu")
learner_net.add_diffeq_node_from_yaml("tanh", node=node, weights=J1, source_var="tanh_op/r",
                                      target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v",
                                      clear=True, float_precision="float64", train_params=["weights"],
                                      node_vars=node_vars)

# train learner net to reproduce the target
n_epochs = 100
inp_epochs = np.tile(inp, (n_epochs, 1, 1))
targets_epoch = np.tile(target, (n_epochs, 1, 1))
train_obs = learner_net.fit_bptt(inp_epochs, targets_epoch, optimizer="rmsprop", retain_graph=True, lr=1e-4)

# simulate fitted network dynamics
fitted_obs = learner_net.run(inp, sampling_steps=1)

# plotting
fig, axes = plt.subplots(nrows=3, figsize=(10, 6))
ax = axes[0]
train_obs.plot("epoch_loss", x="epochs", ax=ax)
ax = axes[1]
fitted_obs.plot("out", ax=ax)
ax.set_title("fitted")
ax = axes[2]
target_obs.plot("out", ax=ax)
ax.set_title("target")
plt.show()
