import matplotlib.pyplot as plt

from rectipy import Network
import numpy as np
from matplotlib.colors import to_hex


# model parameters
node = "neuron_model_templates.spiking_neurons.qif.qif"
N = 100
k = 15.0
tau = 5.0
tau_s = 2.0
eta = -1.0
Delta = 1.0
etas = eta + Delta*np.tan((np.pi/2)*(2*np.arange(1, N+1)-N-1)/(N+1))
v_thr = 100.0
v_reset = -100.0
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
dt = 1e-2
node_vars = {"eta": etas, "tau": tau, "tau_s": tau_s, "k": k}

# initialize target network
target_net = Network(dt, device="cpu")
target_net.add_diffeq_node_from_yaml("qif", node=node, weights=J0, source_var="s", spike_def="v", spike_var="spike",
                                     target_var="s_in", input_var="I_ext", output_var="s", clear=True,
                                     float_precision="float64", op="qif_op", node_vars=node_vars, spike_threshold=v_thr,
                                     spike_reset=v_reset)

# simulate target time series
T = 100.0
steps = int(T/dt)
inp = np.zeros((steps, 1))
time = np.linspace(0, T, steps)
inp[:, 0] = np.sin(2.0*np.pi*0.05*time) * 2.0
target_obs = target_net.run(inp, sampling_steps=1)
target = target_obs.to_numpy("out")

# initialize learner net
J1 = J0 + np.random.randn(N, N)*0.1
learner_net = Network(dt, device="cpu")
learner_net.add_diffeq_node_from_yaml("qif", node=node, weights=J1, source_var="s", spike_def="v", spike_var="spike",
                                      target_var="s_in", input_var="I_ext", output_var="s", clear=True,
                                      float_precision="float64", op="qif_op", train_params=["weights"],
                                      node_vars=node_vars, spike_threshold=v_thr, spike_reset=v_reset)

# train learner net to reproduce the target
n_epochs = 50
inp_epochs = np.tile(inp, (n_epochs, 1, 1))
targets_epoch = np.tile(target, (n_epochs, 1, 1))
train_obs = learner_net.fit_bptt(inp_epochs, targets_epoch, optimizer="rprop", retain_graph=True, lr=1e-3,
                                 optimizer_kwargs={"etas": (0.25, 1.25)}, sampling_steps=1)

# simulate fitted network dynamics
fitted_obs = learner_net.run(inp, sampling_steps=1, verbose=False)

# plotting
neuron_indices = np.arange(0, N, 20)
cmap = plt.get_cmap("plasma", lut=len(neuron_indices))
fig, axes = plt.subplots(nrows=3, figsize=(10, 6))
ax = axes[0]
ax.plot(train_obs["epoch_loss"])
ax = axes[1]
fitted_signal = fitted_obs.get_summary("out")
for i, idx in enumerate(neuron_indices):
    ax.plot(fitted_signal.iloc[:, idx], color=to_hex(cmap(i)))
ax.set_title("fitted")
ax = axes[2]
target_signal = target_obs.get_summary("out")
for i, idx in enumerate(neuron_indices):
    ax.plot(target_signal.iloc[:, idx], color=to_hex(cmap(i)))
ax.set_title("target")
plt.tight_layout()
plt.show()
