from rectipy import Network, random_connectivity
import numpy as np
from matplotlib.pyplot import show

# network parameters
N = 1000
p = 0.1
W = random_connectivity(N, N, p, normalize=True)
eta = -5.0
Delta = 1.0
alpha = 0.0
k = 15.0
etas = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
v_theta = 1e3

# extrinsic input
T = 40.0
dt = 1e-4
steps = int(T/dt)
m = 1
inp = np.zeros((steps, m))
inp[int(10.0/dt):int(30/dt), 0] = 3.0

# initialize network
net = Network(dt, device="cpu")

# add qif node
net.add_diffeq_node("qif", "neuron_model_templates.spiking_neurons.qif.qif_sfa", weights=W, source_var="s",
                    target_var="s_in", input_var="I_ext", output_var="s", spike_def="v", spike_var="spike",
                    op="qif_sfa_op", spike_threshold=v_theta, spike_reset=-v_theta,
                    node_vars={'all/qif_sfa_op/eta': etas, 'all/qif_sfa_op/alpha': alpha, 'all/qif_sfa_op/k': k},
                    float_precision="float64", clear=True)

# add input node
net.add_func_node("inp", m, activation_function="tanh")

# connect input node to qif node
net.add_edge("inp", "qif")

# perform simulation
obs = net.run(inp, record_output=False, record_vars=[("qif", "s", True)], sampling_steps=100)

# plot results
obs.plot(('qif', 's'))
show()
