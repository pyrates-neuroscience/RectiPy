from rectipy import Network, random_connectivity
import numpy as np
from matplotlib.pyplot import show

# network parameters
N = 1000
p = 0.1
W = random_connectivity(N, N, p, normalize=True)
eta = -5.0
Delta = 0.3
etas = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
v_theta = 1e3

# extrinsic input
T = 10.0
dt = 1e-4
steps = int(T/dt)
m = 2
inp = np.random.randn(steps, m)

# initialize network
net = Network.from_yaml("neuron_model_templates.spiking_neurons.qif.qif_sfa_pop", weights=W,
                        source_var="s", target_var="s_in", input_var="I_ext", output_var="s", spike_def="v",
                        spike_var="spike", op="qif_sfa_op", node_vars={'all/qif_sfa_op/eta': etas}, dt=dt,
                        spike_threshold=v_theta, spike_reset=-v_theta, float_precision="float64", record_vars=['s'],
                        clear=True)

# add input and output layers
net.add_input_layer(m, train=False)

# perform simulation
obs = net.run(inp, record_output=False, record_vars=[('s', True)], sampling_steps=100)
net.fit_gd()

# plot results
obs.plot('s')
show()
