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
N = 100
k = 0.9
tau = 10.0
J0 = np.random.randn(N, N)
J0 /= np.max(np.abs(np.linalg.eigvals(J0)))
D = np.random.choice([1.0, 2.0, 3.0], size=(N, N))
S = D*0.3
dt = 1e-3
edge_attr = dict() #{'delay': D, 'spread': S}

# initialize network
net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=J0, dt=dt,
                        edge_attr=edge_attr, source_var="tanh_op/r", target_var="li_op/r_in",
                        input_var="li_op/I_ext", output_var="li_op/v", clear=True, float_precision="float64",
                        node_vars={"all/li_op/k": k, "all/li_op/tau": tau, 'all/li_op/v': np.random.randn(N)},
                        file_name='learning_net', device=device)

# add RLS learning layer and input layer
net.add_input_layer(1)
net.add_output_layer(1, train="rls", beta=0.99, delta=1.0, alpha=1.0)
net.compile()

# online optimization parameters
################################

# meta parameters
steps = 1000000
sample_steps = 100
test_steps = 100000
epsilon = np.float64(0.99)
tol = 1e-5

# input parameters
freq = 0.05
amp = 0.1
W_fb = np.random.randn(N, 1)

# define input
time = np.linspace(0, steps*dt, num=steps)
inp = np.zeros((steps, 1))
target = np.zeros_like(inp)
inp[:, 0] = np.sin(2 * np.pi * freq * time) * amp
target[:, 0] = inp[:, 0]/amp * np.sin(1 * np.pi * freq * time + 0.5*np.pi)

# optimization
##############

obs = net.train_rls(inp, targets=target, feedback_weights=W_fb, update_steps=1000, verbose=True, record_output=True,
                    record_loss=True, tol=tol, loss_beta=epsilon, sampling_steps=sample_steps)
obs.plot("out")
plt.show()

# model testing
###############

print("Starting testing...")
obs2, loss = net.test(inp[:test_steps, :], target[:test_steps], feedback_weights=W_fb, record_output=True,
                      record_loss=False, sampling_steps=1, verbose=False)
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
ax2.plot(target[:test_steps])
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
