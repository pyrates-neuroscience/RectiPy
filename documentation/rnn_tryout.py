import numpy as np
import matplotlib.pyplot as plt
from rectipy import FeedbackNetwork


# build network
###############

# initialize network
dt = 1e-2
net = FeedbackNetwork(dt, "cuda:0")

# add two nodes p1 and p2
N = 100
k = 10.0
neuron = "neuron_model_templates.spiking_neurons.lif.lif"
net.add_diffeq_node("p1", node=neuron, input_var="I_ext", output_var="s", weights=np.random.randn(N, N), source_var="s",
                    target_var="s_in", op="lif_op", spike_var="spike", spike_def="v")
net.add_diffeq_node("p2", node=neuron, input_var="I_ext", output_var="s", weights=np.random.randn(N, N), source_var="s",
                    target_var="s_in", op="lif_op", spike_var="spike", spike_def="v")

# add feedforward edge from p1 to p2
net.add_edge("p1", "p2", weights=k*np.random.rand(N, N), train=None)

# add feedback edge from p2 to p1
net.add_edge("p2", "p1", weights=-10*k*np.random.rand(N, N), feedback=True)

# perform simulation
####################

# define input
steps = 10000
inp = np.zeros((steps, 1)) + 100.0
# inp[4000:6000] += 100.0

# perform simulation
obs = net.run(inputs=inp, sampling_steps=10, enable_grad=False)

# plot results
out = obs.to_numpy("out")
plt.plot(np.mean(out, axis=1))
plt.show()

