from rectipy import Network
from rectipy.ffwd_layer import RLSLayer
import numpy as np
import matplotlib.pyplot as plt
import torch


# preparations
##############

# check CUDA devices
print(f"CUDA available: {torch.cuda.is_available()}")
dev_id = torch.cuda.current_device()
dev_name = torch.cuda.get_device_name(dev_id)
print(f"CUDA device: {dev_name}, ID: {dev_id}")
device = "cuda:0"

# model parameters
node = "model_templates.base_templates.tanh_node"
N = 100
m = 2
k = 0.4
tau = np.random.uniform(10.0, 20.0, size=(N,))
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
W_fb = np.random.randn(N, N)
W_fb /= np.max(np.abs(np.linalg.eigvals(W_fb)))
dt = 1e-2

# initialize network
net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=J0, dt=dt,
                        source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v",
                        clear=True, float_precision="float64", file_name='learning_net', device=device,
                        node_vars={"all/li_op/k": k, "all/li_op/tau": tau, 'all/li_op/v': np.random.randn(N)},
                        )
net.add_input_layer(m)
readout = RLSLayer(N, 1, beta=0.99, alpha=1.0, delta=1.0)
readout.to(device)
net.compile()

# meta parameters
steps = 1000000
test_steps = 100000
sample_steps = 100
rls_steps = 100
epsilon = np.float64(0.99)
beta = np.float64(0.999)
noise = 0.2

# input parameters
f1, f2 = 0.2, 0.02
amp = 0.9

# define input
time = np.linspace(0, steps*dt, num=steps)
inp = np.zeros((steps, m))
target = np.zeros((steps, 1))
inp[:, 0] = np.sin(2 * np.pi * f1 * time) * amp
inp[:, 1] = np.sin(2 * np.pi * f2 * time) * amp
target[:, 0] = inp[:, 0] * inp[:, 1] / amp
inp = torch.tensor(inp, device=device)
target = torch.tensor(target, device=device)

# plt.plot(target)
# plt.show()

# RL procedure
##############

x = torch.zeros((N,), device=device, dtype=torch.float64)
W_fb = torch.tensor(W_fb, device=device)
fb_size = W_fb.shape
CC = torch.tensor(J0, device=device)
out_layer = torch.nn.Tanh()
rl_losses = []
for step in range(steps):

    # forward pass through the rnn layer
    x = out_layer(net.forward(inp[step, :], W_fb @ x))
    y = readout.forward(x)

    # RLS learning of readout weights
    if step % rls_steps == 0:
        readout.update(x, y, target[step, :])

    # RL learning of feedback weights
    rl_loss = 1.0/(1.0 + torch.exp(-readout.loss))
    CC = beta*CC + (1.0 - beta)*torch.outer(x, x)
    diff = CC-W_fb
    W_fb = epsilon*W_fb + (1.0-epsilon)*(rl_loss*torch.randn(fb_size, device=device)*noise + (1.0-rl_loss)*diff)

    # store stuff
    if step % sample_steps == 0:
        rl_losses.append(rl_loss.detach().cpu().numpy())

# testing phase
###############

targets = []
predictions = []
for step in range(test_steps):

    # forward pass through the rnn layer
    x = out_layer(net.forward(inp[step, :], W_fb @ x))
    y = readout.forward(x)

    # store targets and predictions
    targets.append(target[step, 0].detach().cpu().numpy())
    predictions.append(y.detach().cpu().numpy())

# matrices
fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
for ax, mat, title in zip(axes, [W_fb, CC], ["W", "CC"]):
    m = mat.detach().cpu().numpy()
    cax = ax.imshow(m, aspect="equal", interpolation="none")
    plt.colorbar(cax, ax=ax)
    ax.set_title(title)

fig2 = plt.figure("training loss")
ax = fig2.add_subplot()
ax.plot(rl_losses)

fig3 = plt.figure("predictions (test phase)")
ax = fig3.add_subplot()
ax.plot(targets, color="blue", label="targets")
ax.plot(predictions, color="orange", label="predictions")
plt.legend()

plt.show()
