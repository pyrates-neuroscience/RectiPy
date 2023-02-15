from rectipy import Network
from rectipy.optimize import ExpRLS
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
N = 10
m = 3
J = np.random.uniform(low=-1.0, high=1.0, size=(N, N))
D = np.random.choice([1.0, 2.0, 3.0], size=(N, N))
W_out = np.random.randn(m, N)
S = D*0.3
k0 = np.random.uniform(0.1, 10.0)
tau0 = np.random.uniform(0.1, 10.0)
u_idx = np.arange(0, N)

# initialize networks
target_net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=J,
                               edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r", target_var="li_op/r_in",
                               input_var="li_op/I_ext", output_var="li_op/v", clear=True, float_precision="float64",
                               file_name='target_net', node_vars={'all/li_op/v': np.random.randn(N)}, device=device)
target_net.add_output_layer(m, weights=W_out, train=False, activation_function=None)

learning_net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=J,
                                 edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r", target_var="li_op/r_in",
                                 input_var="li_op/I_ext", output_var="li_op/v", clear=False, float_precision="float64",
                                 node_vars={"all/li_op/k": k0, "all/li_op/tau": tau0},
                                 file_name='learning_net', device=device)

# compile networks
target_net.compile()
learning_net.compile()

# create target data
####################

# meta parameters
tol = 1e-5
loss = 1.0
max_steps = 100000
disp_steps = 100
test_steps = 1000
epsilon = 0.999

# input parameters
dt = 1e-3
freq = 0.2
amp = 0.1

# optimization
##############

# optimizer
w = [p for p in learning_net.parameters()]
X_0 = torch.zeros((1, N), device=device, dtype=torch.float64)
y_0 = torch.zeros((1, m), device=device, dtype=torch.float64)
w_0 = torch.zeros((m, N), device=device, dtype=torch.float64)
opt = ExpRLS(X_0, y_0, w_0, beta=0.999, delta=1e-2)

# optimization loop
losses = []
step = 0
while loss > tol and step < max_steps:

    inp = np.sin(2 * np.pi * freq * step * dt) * amp
    y = target_net.forward(inp)
    x = learning_net.forward(inp)
    _ = opt.forward(x, y)
    loss = epsilon*loss + (1-epsilon)*opt.loss.detach().cpu().numpy()
    losses.append(loss)
    step += 1
    print(f"{step} training steps finished. Current loss: {loss}.")

# model testing
###############

# add output layer to learning net
learning_net.add_output_layer(m, weights=opt.w.detach().cpu().numpy(), train=False)
learning_net.compile()

print("Starting testing...")
predictions = []
targets = []
for step in range(test_steps):
    inp = np.sin(2 * np.pi * freq * step * dt) * amp
    prediction = learning_net.forward(inp)
    target = target_net.forward(inp)
    predictions.append(prediction.detach().cpu().numpy())
    targets.append(target.detach().cpu().numpy())
print("Finished.")

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax1 = axes[0]
ax1.plot(predictions)
ax1.set_title('predictions (testing)')
ax1.set_xlabel('steps')
ax1.set_ylabel('u')
ax2 = axes[1]
ax2.plot(targets)
ax2.set_title('targets (testing)')
ax2.set_xlabel('steps')
ax2.set_ylabel('u')
ax3 = axes[2]
ax3.plot(losses)
ax3.set_title('loss (training)')
ax3.set_xlabel('epochs')
ax3.set_ylabel('MSE')
plt.tight_layout()

plt.show()
