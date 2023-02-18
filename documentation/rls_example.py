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

# add RLS learning layer
net.add_output_layer(1, train="rls", beta=0.9, delta=1.0)
net.compile()

# online optimization parameters
################################

# meta parameters
tol = 1e-5
loss = 1.0
max_steps = 1000000
sample_steps = 10
test_steps = 100000
epsilon = np.float64(0.99)

# input parameters
freq = 0.05
amp = 0.1
W_fb = torch.randn(N, 1, device=device, dtype=torch.float64)

# define input
time = np.linspace(0, max_steps*dt, num=max_steps)
inp = np.sin(2 * np.pi * freq * time) * amp
target = inp * np.sin(1 * np.pi * freq * time + 0.5*np.pi) * amp
# plt.plot(time, inp, color="blue", label="input")
# plt.plot(time, target, color="orange", label="target")
# plt.legend()
# plt.show()

# optimization
##############

# optimization loop
losses, train_steps = [], []
step = 0
y_hat = torch.zeros((1,), dtype=torch.float64, device=device)
y_hats = []
while loss > tol and step < max_steps:

    y_hat = net.forward_train_fb(inp[step], target[step], W_fb @ y_hat)
    y_hats.append(y_hat.detach().cpu().numpy())
    step += 1
    if step % sample_steps == 0:
        loss = epsilon * loss + (1.0 - epsilon) * net[-1].loss.detach().cpu().numpy()
        losses.append(loss)
        train_steps.append(step)
        print(f"{step} training steps finished. Current loss: {loss}.")


plt.plot(y_hats)
plt.show()

# model testing
###############

print("Starting testing...")

predictions = []
targets = []
prediction = torch.zeros((1,), dtype=torch.float64, device=device)
for step in range(test_steps):
    prediction = net.forward_fb(inp[step], W_fb @ prediction)
    predictions.append(prediction.detach().cpu().numpy())
    targets.append(target[step])
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
ax3.plot(train_steps, losses)
ax3.set_title('loss (training)')
ax3.set_xlabel('training steps')
ax3.set_ylabel('MSE')
plt.tight_layout()

plt.show()
