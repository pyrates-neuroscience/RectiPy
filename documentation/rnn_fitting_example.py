from rectipy import Network
import numpy as np
import matplotlib.pyplot as plt
import torch


# preparations
##############

# model parameters
node = "model_templates.base_templates.tanh_node"
N = 5
C = np.random.uniform(low=-2.0, high=2.0, size=(N, N))
D = np.random.choice([1.0, 2.0, 3.0], size=(N, N))
#np.save("C.npy", C)
#np.save("D.npy", D)
S = D*0.3

# initialize networks
init_target = np.random.rand(N)
target_net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh_node", weights=C,
                               edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r", target_var="li_op/r_in",
                               input_var_ext="li_op/I_ext", output_var="li_op/u", clear=True, float_precision="float64",
                               node_vars={"all/li_op/u": init_target}, file_name='target_net')
init_learning = np.random.rand(N)
learning_net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh_node", weights=np.ones_like(C),
                                 edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r", target_var="li_op/r_in",
                                 input_var_ext="li_op/I_ext", output_var="li_op/u", clear=True, train_params=['weight'],
                                 float_precision="float64", node_vars={"all/li_op/u": init_learning}, file_name='learning_net')

# compile networks
target_net.compile()
learning_net.compile()

# get target connectivity matrix and connectivity mask
C_target = target_net.rnn_layer.args[-1]
C_mask = torch.abs(C_target) < 1e-5

# model optimization
####################

tol = 1e-2
alpha = 0.9
error = 100.0
dt = 1e-2
step = 0
freq = 0.2
amp = 0.1
update_steps = 10
disp_steps = 500
max_steps = 5000
loss = torch.nn.MSELoss()
opt = torch.optim.SGD(learning_net.parameters(), lr=1e-2)
predictions, targets, losses = [], [], []
while error > tol and step < max_steps:

    inp = np.sin(2*np.pi*freq*step*dt) * amp
    target = target_net.forward(inp)
    prediction = learning_net.forward(inp)
    error_tmp = loss(prediction, target)
    error_tmp.backward(retain_graph=True)
    learning_net.rnn_layer.train_params[0].grad[C_mask] = 0.0
    if step % update_steps == 0:
        opt.step()
        opt.zero_grad()
    if step % disp_steps == 0:
        print(f"Steps finished: {step}. Current loss: {error}")
    step += 1
    error = alpha*error + (1-alpha)*error_tmp.item()
    predictions.append(prediction.detach().numpy())
    targets.append(target.detach().numpy())
    losses.append(error)

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax1 = axes[0]
ax1.plot(predictions)
ax1.set_title('predictions')
ax1.set_xlabel('steps')
ax1.set_ylabel('u')
ax2 = axes[1]
ax2.plot(predictions)
ax2.set_title('targets')
ax2.set_xlabel('steps')
ax2.set_ylabel('u')
ax3 = axes[2]
ax3.plot(losses)
ax3.set_title('loss')
ax3.set_xlabel('steps')
ax3.set_ylabel('MSE')
plt.tight_layout()

fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax1 = axes[0]
ax1.imshow(C_target, aspect='auto')
ax1.set_title("target coupling")
ax2 = axes[1]
ax2.imshow(learning_net.rnn_layer.train_params[0].detach().numpy(), aspect='auto')
ax2.set_title("fitted coupling")
plt.tight_layout()

plt.show()
