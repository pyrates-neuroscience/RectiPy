import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from rectipy import Network
import torch
import torch.nn as nn

device = "cuda:0"
dtype = torch.float64

# network size
nb_inputs = 100
nb_hidden = 4
nb_outputs = 2

# simulation parameters
time_step = 1e-3
nb_steps = 200
nb_epochs = 256

# training data
freq = 5 # Hz
prob = freq*time_step
mask = torch.rand((nb_epochs, nb_steps, nb_inputs), device=device, dtype=dtype)
x_data = torch.zeros((nb_epochs, nb_steps, nb_inputs), device=device, dtype=dtype, requires_grad=False)
x_data[mask < prob] = 1.0

# model parameters
tau_mem = 10e-3
tau_syn = 5e-3
eta = 0.0
v_thr = 1.0
v_reset = 0.0
J0 = np.zeros((nb_hidden, nb_hidden))

weight_scale = 7*(1.0-float(np.exp(-time_step/tau_mem))) # this should give us some spikes to begin with

w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))
node_vars = {"eta": eta, "tau": tau_mem, "tau_s": tau_syn, "k": 1.0}

# model definition
node = "neuron_model_templates.spiking_neurons.lif.lif"
net = Network(time_step, device=device)
net.add_diffeq_node_from_yaml("lif", node=node, weights=J0, source_var="s", spike_def="v", spike_var="spike",
                              target_var="s_in", input_var="I_ext", output_var="s", clear=True,
                              float_precision="float64", op="lif_op", node_vars=node_vars, spike_threshold=v_thr,
                              spike_reset=v_reset)
net.add_func_node(label="inp", n=nb_inputs, activation_function="identity")
net.add_edge("inp", "lif", weights=w1, train="gd")
net.add_func_node(label="out", n=nb_outputs, activation_function="softmax")
net.add_edge("lif", "out", train="gd", weights=w2)

# model training
params = [w1, w2]  # The paramters we want to optimize
optimizer = torch.optim.Adam(params, lr=2e-3, betas=(0.9, 0.999))  # The optimizer we are going to use
loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

# The optimization loop
loss_hist = []
for e in range(nb_epochs):

    # run the network and get output
    obs = net.run(x_data[e], sampling_steps=1, enable_grad=True, verbose=False)
    output = obs["out"]

    # define targets
    y_data = torch.tensor(1 * (np.random.rand(nb_steps) < 0.5), device=device)

    # compute the loss
    log_p_y = torch.log(torch.stack(output))
    loss_val = loss_fn(log_p_y, y_data)

    # update the weights
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    # reset network
    net.reset()

    # store loss value
    loss_hist.append(loss_val.item())
    print(f"Epoch #{e} finished. Epoch loss: {loss_hist[-1]}")

plt.figure(figsize=(3.3, 2), dpi=150)
plt.plot(loss_hist, label="Surrogate gradient")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
sns.despine()
plt.show()
