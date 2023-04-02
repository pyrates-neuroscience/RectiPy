from rectipy import Network
import numpy as np
from matplotlib.pyplot import show


# model parameters
node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
N = 200
m = 2
k = 0.1
tau = np.random.uniform(10.0, 20.0, size=(N,))
eta = 2.0
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
dt = 1e-2

# initialize target network
target_net = Network(dt, device="cuda:0")
target_net.add_diffeq_node_from_yaml("tanh", node=node, weights=J0, source_var="tanh_op/r",
                                     target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v",
                                     clear=True, float_precision="float64")

# simulate target time series
T = 100.0
steps = int(T/dt)
inp = np.zeros((steps, 1))
time = np.linspace(0, T, steps)
inp[:, 0] = np.sin(2.0*np.pi*0.2*time) * 2.0
target_obs = target_net.run(inp, sampling_steps=1)
target = target_obs["out"]

# initialize learner net
J1 = np.random.randn(N, N)
J1 /= np.max(np.abs(np.linalg.eigvals(J1)))
learner_net = Network(dt, device="cuda:0")
learner_net.add_diffeq_node_from_yaml("tanh", node=node, weights=J1, source_var="tanh_op/r",
                                      target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v",
                                      clear=True, float_precision="float64", train_params=["weights"])

# train learner net to reproduce the target
n_epochs = 100
inp_epochs = np.tile(inp, (n_epochs, 1, 1))
targets_epoch = np.tile(target, (n_epochs, 1, 1))
obs = learner_net.fit_bptt(inp_epochs, targets_epoch, optimizer="rmsprop", retain_graph=True)
