from rectipy import Network
import numpy as np
import matplotlib.pyplot as plt
import torch


# preparations
##############

# model parameters
node = "model_templates.base_templates.tanh_node"
N = 5
C = np.load("C.npy")  #np.random.uniform(low=-1.0, high=1.0, size=(N, N))
D =  np.load("D.npy")  #np.random.choice([1.0, 2.0, 3.0], size=(N, N))
#np.save("C.npy", C)
#np.save("D.npy", D)
S = D*0.3
J0 = np.random.uniform(0.1, 10.0)
tau0 = np.random.uniform(0.1, 10.0)
u_idx = np.arange(0, N)

# initialize networks
target_net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh_node", weights=C,
                               edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r", target_var="li_op/r_in",
                               input_var_ext="li_op/I_ext", output_var="li_op/u", clear=True, float_precision="float64",
                               file_name='target_net', node_vars={'all/li_op/u': np.random.randn(N)})

learning_net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh_node", weights=C,
                                 edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r", target_var="li_op/r_in",
                                 input_var_ext="li_op/I_ext", output_var="li_op/u", clear=False,
                                 train_params=['li_op/J', 'li_op/tau'], float_precision="float64",
                                 node_vars={"all/li_op/J": J0, "all/li_op/tau": tau0},
                                 file_name='learning_net')

# compile networks
target_net.compile()
learning_net.compile()

# extract initial value vector for later state vector resets
y0 = target_net.rnn_layer.y.clone().detach().numpy()

# create target data
####################

# error parameters
tol = 1e-3
error = 1.0

# input parameters
dt = 1e-3
freq = 0.2
amp = 0.1

# epoch parameters
n_epochs = 100
disp_steps = 1000
epoch_steps = 30000
epoch = 0

# target data creation
print("Creating target data...")
target_net.rnn_layer.reset(y0)
targets = []
for step in range(epoch_steps):
    inp = np.sin(2 * np.pi * freq * step * dt) * amp
    target = target_net.forward(inp)
    targets.append(target)
print("Finished.")

plt.plot([t.detach().numpy() for t in targets])
plt.show()

# optimization
##############

# loss and optimizer
loss = torch.nn.MSELoss()
opt = torch.optim.Rprop(learning_net.parameters(), lr=0.01, etas=(0.5, 1.1), step_sizes=(1e-5, 1e-1))

# optimization loop
losses = []
while error > tol and epoch < n_epochs:

    # error calculation epoch
    losses_tmp = []
    learning_net.rnn_layer.reset(y0)
    for step in range(epoch_steps):
        inp = np.sin(2*np.pi*freq*step*dt) * amp
        target = targets[step]
        prediction = learning_net.forward(inp)
        error_tmp = loss(prediction, target)
        error_tmp.backward(retain_graph=True)
        if step % disp_steps == 0:
            print(f"Steps finished: {step}. Current loss: {error_tmp.item()}")
        losses_tmp.append(error_tmp.item())

    # optimization step
    opt.step()
    opt.zero_grad()
    error = np.mean(losses_tmp)
    losses.append(error)
    epoch += 1
    print(f"Training epoch #{epoch} finished. Mean epoch loss: {error}.")

# model testing
###############

print("Starting testing...")
learning_net.rnn_layer.reset(y0)
predictions = []
for step in range(epoch_steps):
    inp = np.sin(2 * np.pi * freq * step * dt) * amp
    prediction = learning_net.forward(inp)
    predictions.append(prediction.detach().numpy())
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
ax2.plot([t.detach().numpy() for t in targets])
ax2.set_title('targets (testing)')
ax2.set_xlabel('steps')
ax2.set_ylabel('u')
ax3 = axes[2]
ax3.plot(losses)
ax3.set_title('loss (training)')
ax3.set_xlabel('epochs')
ax3.set_ylabel('MSE')
plt.tight_layout()

for key, target, val, start in zip(["J", "tau"], [target_net.rnn_layer.args[1].numpy(), target_net.rnn_layer.args[0].numpy()],
                                   [learning_net.rnn_layer.args[1].detach().numpy(), learning_net.rnn_layer.args[0].detach().numpy()],
                                   [J0, tau0]):
    print(f"Parameter: {key}. Target: {target}. Fitted value: {val}. Initial value: {start}.")

plt.show()
