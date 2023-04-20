import matplotlib.pyplot as plt
import torch
from torch import nn
from rectipy import Network
import numpy as np

device = "cpu"
dtype = torch.float64

# model parameters
node = "neuron_model_templates.spiking_neurons.lif.lif"
N = 100
k = 2.0
tau = np.random.uniform(10.0, 15.0, size=(N,))
tau_s = 5.0
eta = 10.0
v_thr = 10.0
v_reset = -10.0
J0 = np.zeros((N, N))
dt = 1e-2
node_vars = {"eta": eta, "tau": tau, "tau_s": tau_s, "k": k}

# define input and output layer dimensions
n_in = 2
n_out = 3
W_in = np.random.randn(N, n_in)
W_out = np.random.randn(n_out, N)
W_in_0 = np.random.randn(N, n_in)
W_out_0 = np.random.randn(n_out, N)

# initialize target network
net = Network(dt, device=device)
net.add_diffeq_node("lif", node=node, weights=J0, source_var="s", spike_def="v", spike_var="spike", target_var="s_in",
                    input_var="I_ext", output_var="s", clear=True, float_precision="float64", op="lif_op",
                    node_vars=node_vars, spike_threshold=v_thr, spike_reset=v_reset)
net.add_func_node(label="inp", n=n_in, activation_function="identity")
net.add_edge("inp", "lif", weights=W_in)
net.add_func_node(label="out", n=n_out, activation_function="identity")
net.add_edge("lif", "out", weights=W_out)
net.compile()

# initialize learner network
learner_net = Network(dt, device=device)
learner_net.add_diffeq_node("lif", node=node, weights=J0, source_var="s", spike_def="v", spike_var="spike",
                            target_var="s_in", input_var="I_ext", output_var="s", clear=True, float_precision="float64",
                            op="lif_op", node_vars=node_vars, spike_threshold=v_thr, spike_reset=v_reset)
learner_net.add_func_node(label="inp", n=n_in, activation_function="identity")
learner_net.add_edge("inp", "lif", train="gd", weights=W_in_0)
learner_net.add_func_node(label="out", n=n_out, activation_function="identity")
learner_net.add_edge("lif", "out", train="gd", weights=W_out_0)
learner_net.compile()

# define training parameters
T = 100.0
steps = int(T/dt)
epochs = 100
inputs = torch.zeros((steps, n_in), dtype=dtype)
omegas = [0.03, 0.05]
time = torch.linspace(0, T, steps=steps)
for idx, omega in enumerate(omegas):
    inputs[:, idx] = torch.sin(time*2.0*np.pi*omega)
optimizer = torch.optim.Rprop(learner_net.parameters(), lr=0.05, etas=(0.5, 1.1), step_sizes=(1e-6, 0.9))
loss_fn = nn.MSELoss()

# get targets
obs = net.run(inputs, sampling_steps=1, enable_grad=False, verbose=False)
targets = torch.stack(obs["out"])

# get initial dynamics of learnet network
obs_learner = learner_net.run(inputs, sampling_steps=1, enable_grad=False, verbose=False)
predictions_init = obs_learner.to_numpy("out")

# fit learner network
loss_hist = []
print_steps = 100
update_steps = 100
loss_val = 1.0
for epoch in range(epochs):

    # perform forward pass
    obs_learner = learner_net.run(inputs, sampling_steps=1, enable_grad=True, verbose=False)
    predictions = obs_learner["out"]

    # calculate loss
    loss = loss_fn(torch.stack(predictions), targets)

    # perform weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    learner_net.reset()

    # save stuff for plotting
    loss_hist.append(loss.item())
    print(f"Epoch #{epoch} finished. Current loss = {loss_hist[-1]}")

# extract fitted weights of learner network
params = list(learner_net.parameters())
W_in_1 = params[-1].detach().cpu().numpy()
W_out_1 = params[0].detach().cpu().numpy()

# get predictions
obs_learner = learner_net.run(inputs, sampling_steps=1, enable_grad=False, verbose=False)
predictions = obs_learner.to_numpy("out")

# plotting
fig, axes = plt.subplots(nrows=4, figsize=(10, 8))
ax = axes[0]
ax.plot(loss_hist)
ax.set_title("Loss")
ax.set_xlabel("step")
ax.set_ylabel("MSE")
targets = obs.to_numpy("out")
for idx in range(n_out):
    ax = axes[idx+1]
    ax.plot(predictions_init[:, idx], label="initial guess")
    ax.plot(predictions[:, idx], label="prediction")
    ax.plot(targets[:, idx], label="target")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("out")
plt.tight_layout()

fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(10, 10))
ax = axes[0, 0]
im = ax.imshow(W_in_0, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Original W_in")
ax = axes[0, 1]
im = ax.imshow(W_out_0, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Original W_out")
ax = axes[1, 0]
im = ax.imshow(W_in_1, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Fitted W_in")
ax = axes[1, 1]
im = ax.imshow(W_out_1, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Fitted W_out")
ax = axes[2, 0]
im = ax.imshow(W_in_0 - W_in_1, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Changes to W_in")
ax = axes[2, 1]
im = ax.imshow(W_out_0 - W_out_1, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Changes to W_out")
ax = axes[3, 0]
im = ax.imshow(W_in, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Target W_in")
ax = axes[3, 1]
im = ax.imshow(W_out, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Target W_out")
plt.tight_layout()

plt.show()
