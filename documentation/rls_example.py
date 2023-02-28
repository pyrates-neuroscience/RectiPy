from rectipy import Network
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
N = 600
m = 2
k = 0.8
tau = np.random.uniform(10.0, 20.0, size=(N,))
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
dt = 1e-2

# initialize network
net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=J0, dt=dt,
                        source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v",
                        clear=True, float_precision="float64", file_name='learning_net', device=device,
                        node_vars={"all/li_op/k": k, "all/li_op/tau": tau, 'all/li_op/v': np.random.randn(N)},
                        )

# add RLS learning layer and input layer
net.add_input_layer(m)
net.add_edge(1, train="rls", beta=0.99, delta=1.0, alpha=1.0)
net.compile()

# online optimization parameters
################################

# meta parameters
steps = 1000000
sample_steps = 1000
test_steps = 20000
epsilon = np.float64(0.99)
tol = 1e-4

# input parameters
f1, f2 = 0.2, 0.02
amp = 1.5
W_fb = np.random.randn(N, 1)

# define input
time = np.linspace(0, steps*dt, num=steps)
inp = np.zeros((steps, m))
target = np.zeros((steps, 1))
inp[:, 0] = np.sin(2 * np.pi * f1 * time) * amp
inp[:, 1] = np.sin(2 * np.pi * f2 * time) * amp
target[:, 0] = inp[:, 0] * inp[:, 1] / amp

# plt.plot(target)
# plt.show()

# optimization
##############

obs = net.fit_rls(inp, targets=target, update_steps=100, verbose=True, record_output=True, record_loss=True,
                  tol=tol, loss_beta=epsilon, sampling_steps=sample_steps, feedback_weights=W_fb)
obs.plot("out")
plt.show()

# model testing
###############

print("Starting testing...")
ds_test = 10
obs2, loss = net.test(inp[:test_steps, :], target[:test_steps], record_output=True, record_loss=False,
                      sampling_steps=ds_test, verbose=False, feedback_weights=W_fb)
print("Finished.")

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax1 = axes[0]
obs2.plot("out", ax=ax1)
ax1.set_title('predictions (testing)')
ax1.set_xlabel('test steps')
ax1.set_ylabel('u')
ax2 = axes[1]
ax2.plot(target[:test_steps:ds_test])
ax2.set_title('targets (testing)')
ax2.set_xlabel('test steps')
ax2.set_ylabel('u')
ax3 = axes[2]
obs.plot("loss", ax=ax3)
ax3.set_title('loss (training)')
ax3.set_xlabel('training steps')
ax3.set_ylabel('MSE')
plt.tight_layout()

plt.show()
