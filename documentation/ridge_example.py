from rectipy import Network, random_connectivity, input_connections, wta_score, readout
import numpy as np
import matplotlib.pyplot as plt


# preparations
##############

# network parameters
N = 1000
p = 0.06
eta = 0.0
Delta = 0.1
etas = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
v_theta = 1e3
Delta_in = 4.0
J = 20.0

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
T_init = 50.0
T_syll = 0.5
n_syll = len(s1)
n_reps = 100
T_epoch = T_syll*n_syll*n_reps
dt = 1e-3
n_epochs = 2
train_epochs = 1

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
net = Network.from_yaml("neuron_model_templates.spiking_neurons.qif.qif_sfa_pop", weights=W*J,
                        source_var="s", target_var="s_in", input_var="I_ext", output_var="s", spike_def="v",
                        spike_var="spike", op="qif_sfa_op", node_vars={'all/qif_sfa_op/eta': etas}, dt=dt,
                        spike_threshold=v_theta, spike_reset=-v_theta, float_precision="float64", clear=True)

# wash out initial condition
net.run(np.zeros((init_steps, 1)), verbose=False, sampling_steps=init_steps+1)

# add input layer
net.add_input_layer(m, weights=W_in, train=False)
net.compile()

coeffs = []
for j in range(train_epochs):

    # gather data
    obs = net.run(inp[j], sampling_steps=1, verbose=False, record_output=True)
    X = obs['out']

    # train readout layer weights
    _, coeffs_tmp = readout(X, targets[j], fit_intercept=False, copy_X=True)
    coeffs.append(coeffs_tmp)

# add output layers
net.add_output_layer(k, train=False, weights=np.mean(coeffs, axis=0))
net.compile()

# test performance on last epoch
obs, test_loss = net.test(inp[train_epochs], targets[train_epochs], loss='ce', record_output=True, sampling_steps=1,
                          verbose=False)

# calculate WTA score
wta = wta_score(np.asarray(obs['out']), targets[train_epochs])
print(f'Finished. Loss on test data set: {test_loss}. WTA score: {wta}.')

# plot predictions vs. targets
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.argmax(obs['out'], axis=-1))
ax.plot(np.argmax(targets[train_epochs], axis=-1))
plt.show()
