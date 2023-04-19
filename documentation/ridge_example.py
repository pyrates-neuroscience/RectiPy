from rectipy import Network, random_connectivity, input_connections, wta_score
import numpy as np
import matplotlib.pyplot as plt


# preparations
##############

# network parameters
N = 100
p = 0.1
Delta_in = 2.0
J = 2.0

# input parameters
m = 5
p_in = 0.5
s1 = [0, 2, 1]
s2 = [4, 2, 3]
s3 = [3, 2, 0]
signals = [s1, s2, s3]

# output parameters
k = len(signals)

# training parameters
T_init = 100.0
T_syll = 1.0
n_syll = len(s1)
n_reps = 100
T_epoch = T_syll*n_syll*n_reps
dt = 1e-3
n_epochs = 21
train_epochs = n_epochs-1

# define extrinsic input and targets
epoch_steps = int(T_epoch/dt)
syll_steps = int(T_syll/dt)
init_steps = int(T_init/dt)
inp = np.zeros((n_epochs, epoch_steps, m))
targets = np.zeros((n_epochs, epoch_steps, k))
for epoch in range(n_epochs):
    for rep in range(n_reps):
        choice = np.random.choice(k)
        s = signals[choice]
        for idx in range(n_syll):
            inp[epoch, (rep*n_syll+idx)*syll_steps:(rep*n_syll+idx+1)*syll_steps, s[idx]] = 1.0
        targets[epoch, rep*n_syll*syll_steps:(rep+1)*n_syll*syll_steps, choice] = 1.0

# generate connectivity matrix
W = random_connectivity(N, N, p, normalize=True)

# generate input matrix
W_in = input_connections(N, m, p_in, variance=Delta_in, zero_mean=True)

# optimization
##############

# initialize network
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("tanh", "neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=W*J,
                    source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v",
                    float_precision="float64", clear=True)

# wash out initial condition
net.run(np.zeros((init_steps, 1)), verbose=False, sampling_steps=init_steps+1)

# add input layer
net.add_func_node("inp", m, activation_function="identity")
net.add_edge("inp", "tanh", weights=W_in)
net.compile()

coeffs = []
for j in range(train_epochs):

    # train on epoch
    obs = net.fit_ridge(inputs=inp[j], targets=targets[j], sampling_steps=1, verbose=False, add_readout_node=False,
                        alpha=1e-4)

    # store readout weights
    coeffs.append(obs.to_numpy("w_out"))

    # display progress
    print(f"Epoch #{j+1} finished.")

# add output layers
w_out = np.mean(coeffs, axis=0)
net.add_func_node("readout", k, activation_function="identity")
net.add_edge("tanh", "readout", weights=w_out)
net.compile()

# test performance on last epoch
obs, test_loss = net.test(inp[train_epochs], targets[train_epochs], loss='mse', record_output=True, sampling_steps=1,
                          verbose=False)

# calculate WTA score
wta = wta_score(obs.to_numpy("out"), targets[train_epochs])
print(f'Finished. Loss on test data set: {test_loss}. WTA score: {wta}.')

# plot predictions vs. targets
fig, axes = plt.subplots(nrows=k, figsize=(12, 9))
predictions = obs.to_numpy("out")
for i, ax in enumerate(axes):
    ax.plot(targets[train_epochs, :, i], "blue", label="target")
    ax.plot(predictions[:, i], "orange", label="prediction")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("out")
plt.show()
