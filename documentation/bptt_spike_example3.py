import matplotlib.pyplot as plt
import torch
from torch import nn
from rectipy import Network
import numpy as np
from matplotlib.colors import to_hex

device = "cpu"

# model parameters
node = "neuron_model_templates.spiking_neurons.lif.lif"
N = 200
k = 5.0
tau = np.random.uniform(1.0, 5.0, size=(N,))
tau_s = 2.0
eta = 5.0
v_thr = 100.0
v_reset = -100.0
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
dt = 2e-3
node_vars = {"eta": eta, "tau": tau, "tau_s": tau_s, "k": k}

# define input layer
m = 2
W_in = np.random.randn(N, m)

# initialize target network
net = Network(dt, device=device)
net.add_diffeq_node_from_yaml("lif", node=node, weights=J0, source_var="s", spike_def="v", spike_var="spike",
                              target_var="s_in", input_var="I_ext", output_var="s", clear=True,
                              float_precision="float64", op="lif_op", node_vars=node_vars, spike_threshold=v_thr,
                              spike_reset=v_reset, train_params=["weights"])
net.add_func_node(label="inp", n=m, activation_function="identity")
net.add_edge("inp", "lif", weights=W_in)
net.add_func_node(label="out", n=m, activation_function="softmax")
net.add_edge("lif", "out", train="gd")

# define epoch parameters
T = 100.0
steps = int(T/dt)
in_dur = 10.0
in_steps = int(in_dur/dt)
n_epochs = 100

# train learner net to reproduce the target
optimizer = torch.optim.Adamax(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss_fn = nn.NLLLoss()

loss_hist = []
for e in range(n_epochs):

    # define input
    targets = torch.zeros((steps,), dtype=torch.int64, device=device)
    inp = torch.zeros((steps, m), dtype=torch.float64, device=device, requires_grad=False)
    step = 0
    while step < steps - 1:
        mus = torch.rand(2)*20.0
        sigmas = mus*0.2
        for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
            inp[step:step+in_steps, i] = mu + torch.randn(in_steps, device=device, requires_grad=False) * sigma
        targets[step:step+in_steps] = 0 if mus[0] > mus[1] else 1
        step += in_steps

    # train network
    obs = net.run(inputs=inp, sampling_steps=1, verbose=False, enable_grad=True, record_vars=[("lif", "s", False)])
    output = obs["out"]
    log_p_y = torch.log(torch.stack(output))
    loss_val = loss_fn(log_p_y, targets)

    loss_val.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_hist.append(loss_val.item())

    # plotting
    fig, axes = plt.subplots(nrows=3, figsize=(10, 6))
    ax = axes[0]
    ax.plot(loss_hist)
    ax.set_title("Epoch loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax = axes[1]
    signal = obs.to_numpy("out")
    ax.plot(np.argmax(signal, axis=1), label="fitted")
    ax.plot(targets.detach().cpu().numpy(), label="target")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("input class")
    ax.set_title(f"Epoch loss = {loss_hist[-1]}")
    ax = axes[2]
    obs.plot(("lif", "s"), ax=ax)
    plt.tight_layout()
    plt.show()

    print(f"Epoch #{e} finished. Epoch loss = {loss_hist[-1]}")
    net.reset()

# simulate fitted network dynamics
targets = torch.zeros((steps,), dtype=torch.int64, device=device)
inp = torch.zeros((steps, m), dtype=torch.float64, device=device)
step = 0
while step < steps - 1:
    mus = torch.rand(2)*5.0
    sigmas = mus*0.2
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        inp[step:step+in_steps, i] = mu + torch.randn(in_steps) * sigma
    targets[step:step+in_steps] = 1 if mus[0] > mus[1] else 2
    step += in_steps
fitted_obs = net.run(inp, sampling_steps=1, verbose=False, enable_grad=True)
log_p_y = torch.log(torch.stack(fitted_obs["out"]))
test_loss = loss_fn(log_p_y, targets).item()

# plotting
fig, axes = plt.subplots(nrows=2, figsize=(10, 6))
ax = axes[0]
ax.plot(loss_hist)
ax.set_title("Epoch loss")
ax.set_xlabel("epoch")
ax.set_ylabel("NLL")
ax = axes[1]
fitted_signal = fitted_obs.get_summary("out")
ax.plot(np.argmax(fitted_signal, axis=1), label="fitted")
ax.plot(targets, label="target")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("input class")
ax.set_title(f"Test data loss = {test_loss}")
plt.tight_layout()
plt.show()
