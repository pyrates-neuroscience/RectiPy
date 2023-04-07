import matplotlib.pyplot as plt
import torch
from torch import nn
from rectipy import Network
import numpy as np
from matplotlib.colors import to_hex


# model parameters
node = "neuron_model_templates.spiking_neurons.lif.lif"
N = 200
k = 2.0
tau = np.random.uniform(10.0, 20.0, size=(N,))
tau_s = 5.0
eta = 10.0
v_thr = 100.0
v_reset = -100.0
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
dt = 1e-2
node_vars = {"eta": eta, "tau": tau, "tau_s": tau_s, "k": k}

# initialize target network
target_net = Network(dt, device="cpu")
target_net.add_diffeq_node_from_yaml("lif", node=node, weights=J0, source_var="s", spike_def="v", spike_var="spike",
                                     target_var="s_in", input_var="I_ext", output_var="s", clear=True,
                                     float_precision="float64", op="lif_op", node_vars=node_vars, spike_threshold=v_thr,
                                     spike_reset=v_reset
                                     )

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
learner_net = Network(dt, device="cpu")
learner_net.add_diffeq_node_from_yaml("lif", node=node, weights=J1, source_var="s", spike_def="v", spike_var="spike",
                                      target_var="s_in", input_var="I_ext", output_var="s", clear=True,
                                      float_precision="float64", op="lif_op", train_params=["weights"],
                                      node_vars=node_vars, spike_threshold=v_thr, spike_reset=v_reset)

# train learner net to reproduce the target
optimizer = torch.optim.Rprop(learner_net.parameters(), lr=1e-2, etas=(0.5, 1.1), step_sizes=(1e-6, 1e-1))
loss_fn = nn.MSELoss()

loss_hist = []
for e in range(100):
    output = learner_net.run(inputs=inp, sampling_steps=1, verbose=False, enable_grad=True)["out"]
    loss_val = loss_fn(torch.stack(output), torch.stack(target))

    loss_val.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_hist.append(loss_val.item())

    print(f"Epoch #{e} finished. Epoch loss = {loss_hist[-1]}")
    learner_net.reset()

# simulate fitted network dynamics
fitted_obs = learner_net.run(inp, sampling_steps=1, verbose=False)

# plotting
neuron_indices = np.arange(0, N, 50)
cmap = plt.get_cmap("plasma", lut=len(neuron_indices))
fig, axes = plt.subplots(nrows=3, figsize=(10, 6))
ax = axes[0]
ax.plot(loss_hist)
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
