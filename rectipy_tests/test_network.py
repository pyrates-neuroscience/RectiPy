"""Test suite for the network functionalities, which serves as the main user interface.
"""

# imports
from rectipy.nodes import RateNet, SpikeNet, InstantNode
from rectipy.edges import Linear, RLS
from rectipy import Network
import torch
import pytest
import numpy as np
from pyrates import clear_frontend_caches, CircuitTemplate, NodeTemplate


# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


# Utility
#########


def setup_module():
    print("\n")
    print("========================")
    print("| Test Suite : Network |")
    print("========================")


# test accuracy
accuracy = 1e-3


# tests
#######


def test_3_1_diffeq_nodes():
    """Tests Network whether differential-equation-based nodes behave as expected.
    """

    clear_frontend_caches()

    # parameters
    n = 10
    dt = 1e-2
    weights = np.random.randn(n, n)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    node_spiking = "neuron_model_templates.spiking_neurons.qif.qif"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # RNN layer initialization
    rnn = RateNet.from_pyrates(node, weights=weights, source_var=s_var, target_var=t_var, input_var=in_var,
                               output_var=out_var, clear=True, verbose=False, dt=dt)

    # RNN template initialization
    node_temp = NodeTemplate.from_yaml(node)
    nodes = {f"p{i}": node_temp for i in range(weights.shape[0])}
    circ_temp = CircuitTemplate("tanh_net", nodes=nodes)
    circ_temp.add_edges_from_matrix(source_var=s_var, target_var=t_var, weight=weights, source_nodes=list(nodes.keys()))

    # different network initializations with a single RNN layer
    net1, net2, net3, net4, net5, net6 = Network(dt), Network(dt), Network(dt), Network(dt), Network(dt), Network(dt)
    net1.add_diffeq_node("n1", node=node, input_var=in_var, output_var=out_var, weights=weights, source_var=s_var,
                         target_var=t_var, clear=True, verbose=False)
    net2.add_diffeq_node("n1", node=circ_temp, input_var=in_var, output_var=out_var, source_var=s_var, target_var=t_var,
                         clear=True, verbose=False)
    net3.add_diffeq_node("n1", node, weights=weights, input_var="I_ext", output_var=out_var, source_var=s_var,
                         target_var="r_in", clear=True, verbose=False, op="li_op")
    net4.add_diffeq_node("n1", node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                         target_var=t_var, clear=True, verbose=False, train_params=["weights"])
    net5.add_diffeq_node("n1", node_spiking, weights=weights, input_var="I_ext", output_var="s", source_var="s",
                         target_var="s_in", op="qif_op", spike_var="spike", spike_def="v", clear=True, verbose=False,
                         dtype=torch.float32)
    net6.add_node("n1", rnn, node_type="diff_eq")

    # these tests should pass
    assert isinstance(net1.get_node("n1"), RateNet)
    assert isinstance(net2.get_node("n1"), RateNet)
    assert isinstance(net6.get_node("n1"), RateNet)
    assert isinstance(net5.get_node("n1"), SpikeNet)
    assert isinstance(net1["n1"]["node"], RateNet)
    assert net6.get_node("n1") == rnn
    assert len(net3._var_map) - len(net1._var_map) > 0
    assert len(net1.get_node("n1").train_params) == 0
    assert len(net4.get_node("n1").train_params) == 1
    assert net3.get_var("n1", var="v").shape[0] == n
    assert net1.get_node("n1").y.dtype == torch.float64
    assert net5.get_node("n1").y.dtype == torch.float32

    # these tests should fail
    with pytest.raises(FileNotFoundError):
        RateNet.from_pyrates("neuron_model_templates.rate_neurons.freaky_integrator.tanh", weights=weights,
                             source_var=s_var, target_var=t_var, input_var=in_var, output_var=out_var, clear=True,
                             verbose=False)
    with pytest.raises(AttributeError):
        RateNet.from_pyrates("neuron_model_templates.rate_neurons.leaky_integrator.tan", weights=weights,
                             source_var=s_var, target_var=t_var, input_var=in_var, output_var=out_var, clear=True,
                             verbose=False)
    with pytest.raises(KeyError):
        RateNet.from_pyrates(node, weights=weights, source_var="x", target_var=t_var, input_var=in_var,
                             output_var=out_var, clear=True, verbose=False)


def test_3_2_function_nodes():
    """Tests whether activation function nodes behave as expected in Network.
    """

    # input parameters
    m = 3
    x = torch.randn(m, dtype=torch.float32)
    y = torch.randn(m, dtype=torch.float64)

    # initialize network with different activation function nodes
    net = Network(dt=1e-3)
    net.add_func_node("softmax", m, activation_function="softmax")
    net.add_func_node("sigmoid", m, activation_function="sigmoid")
    net.add_func_node("identity", m, activation_function="softmax")

    # these tests should pass
    assert isinstance(net.get_node("softmax"), InstantNode)
    assert isinstance(net.get_node("sigmoid"), InstantNode)
    assert isinstance(net["softmax"]["node"], InstantNode)
    assert net.get_node("sigmoid").forward(x).shape[0] == m
    assert net.get_node("softmax").forward(x).dtype == torch.float32
    assert net.get_node("softmax").forward(y).dtype == torch.float64
    net.pop_node("softmax")
    assert len(net.nodes) == 2

    # these tests should fail
    with pytest.raises(ValueError):
        net.add_func_node("wrong", m, activation_function="kickmoid")


def test_3_3_edges():
    """Tests whetheredges added to Netowkr instances behave as expected.
    """

    # rnn parameters
    n = 10
    weights = np.random.randn(n, n)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # read parameters
    k = 3
    out_weights = np.random.randn(n, k)

    # input definition
    x = torch.randn(n, dtype=torch.float64)

    # set up network with a single RNN layer and multiple, independent readout layers
    net = Network(dt=1e-3)
    net.add_diffeq_node("rnn", node=node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                        target_var=t_var, clear=True, verbose=False, dtype=torch.float64)
    net.add_func_node("readout_1", k, activation_function="identity")
    net.add_func_node("readout_2", k, activation_function="sigmoid")
    net.add_func_node("readout_3", k, activation_function="identity")
    net.add_func_node("readout_4", k, activation_function="identity")

    # add edges between RNN and readout layers
    net.add_edge("rnn", "readout_1", weights=out_weights, dtype=torch.float64)
    net.add_edge("rnn", "readout_2", dtype=torch.float32)
    net.add_edge("rnn", "readout_3", weights=out_weights, dtype=torch.float64, train="gd")
    net.add_edge("rnn", "readout_4", weights=out_weights, dtype=torch.float64, train="rls")

    # these tests should pass
    assert isinstance(net.get_edge("rnn", "readout_1"), Linear)
    assert isinstance(net.get_edge("rnn", "readout_3"), Linear)
    assert isinstance(net.get_edge("rnn", "readout_4"), RLS)
    assert len(list(net.parameters(recurse=True))) == 1
    assert net.get_edge("rnn", "readout_2").weights.dtype == torch.float32
    assert net.get_edge("rnn", "readout_1").forward(x).shape[0] == k

    # these tests should fail
    net.pop_edge("rnn", "readout_1")
    with pytest.raises(ValueError):
        net.add_edge("rnn", "readout_1", weights=torch.randn(n, k+1))
    with pytest.raises(KeyError):
        net.add_edge("rnn_1", "readout_1", weights=torch.randn(n, k + 1))
    with pytest.raises(RuntimeError):
        net.get_edge("rnn", "readout_2").forward(x)


def test_3_4_compile():
    """Tests compile functionalities of Network class.
    """

    # rnn parameters
    n = 10
    weights = np.random.randn(n, n)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # output parameters
    k = 2

    # input parameters
    m = 3
    x = torch.randn(m, dtype=torch.float64)

    # network initialization
    net = Network(dt=1e-3)
    net.add_diffeq_node('rnn', node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                        target_var=t_var, clear=True, verbose=False, dtype=torch.float64)

    # these tests should pass
    net.compile()
    assert len(net._bwd_graph) == 0
    net.add_func_node("inp", m, activation_function="identity")
    net.add_edge("inp", "rnn", dtype=torch.float64)
    net.compile()
    assert len(net._bwd_graph) == 1
    y1 = net.forward(x)
    net.add_func_node("out", k, activation_function="sigmoid")
    net.add_edge("rnn", "out", dtype=torch.float64)
    net.compile()
    y2 = net.forward(x)
    assert len(net._bwd_graph) == 2
    assert y2.shape[0] - y1.shape[0] == k-n

    # these tests should fail
    net.pop_node("inp")
    net.compile()
    with pytest.raises(RuntimeError):
        net.forward(x)
    net.add_func_node("out2", k, activation_function="sigmoid")
    net.add_edge("rnn", "out2")
    with pytest.raises(ValueError):
        net.compile()


def test_3_5_parameters():
    """Tests parameters method of Network class.
    """

    # parameters
    n = 10
    k = 3
    m = 2
    weights = np.random.randn(n, n)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # network initialization
    net1, net2 = Network(dt=1e-3), Network(dt=1e-3)
    net1.add_diffeq_node("rnn", node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                         target_var=t_var, clear=True, verbose=False, file_name="net1", dtype=torch.float64)
    net2.add_diffeq_node("rnn", node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                         target_var=t_var, clear=True, verbose=False, file_name="net2", dtype=torch.float64,
                         train_params=['weights', 'li_op/tau'])

    # test number of parameters
    assert len(list(net1.parameters())) == 0
    assert len(list(net2.parameters())) == 2

    # add input layers
    net1.add_func_node("inp", m, activation_function="identity")
    net2.add_func_node("inp", m, activation_function="identity")
    net1.add_edge("inp", "rnn", train="gd")
    net2.add_edge("inp", "rnn", train=None)
    net1.compile()
    net2.compile()

    # test number of parameters
    assert len(list(net1.parameters())) == 1
    assert len(list(net2.parameters())) == 2

    # add output layers
    net1.add_func_node("out", k, activation_function="identity")
    net2.add_func_node("out", k, activation_function="identity")
    net1.add_edge("rnn", "out", train="gd")
    net2.add_edge("rnn", "out", train="rls")
    net1.compile()
    net2.compile()

    # test number of parameters
    assert len(list(net1.parameters())) == 2
    assert len(list(net2.parameters())) == 2


def test_3_6_simulation():
    """Tests simulation functionalities of Network class.
    """

    # rnn parameters
    dt = 1e-2
    n = 10
    steps = 100
    weights = np.random.randn(n, n)
    x = torch.randn(steps, n, dtype=torch.float64)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # network initialization
    net1, net2, net3 = Network(dt=dt), Network(dt=dt), Network(dt=dt)
    net1.add_diffeq_node("rnn", node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                         target_var=t_var, file_name="net1", dtype=torch.float64)
    net2.add_diffeq_node("rnn", node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                         target_var=t_var, file_name="net2", dtype=torch.float64,
                         record_vars=['li_op/v'])
    net3.add_diffeq_node("rnn", node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                         target_var=t_var, file_name="net3", dtype=torch.float64,
                         record_vars=['li_op/v']
                         )
    net3.compile()

    # run simulations
    res1 = net1.run(inputs=x, sampling_steps=2, verbose=False)
    res2 = net2.run(inputs=x, record_output=False, record_vars=[('rnn', 'li_op/v', False)], verbose=False)
    res3, res4 = [], []
    for step in range(steps):
        out = net3.forward(x[step, :])
        if step % 2 == 0:
            res3.append(out.detach().numpy())
        res4.append(net3.get_var("rnn", var="li_op/v").detach().numpy())

    # these tests should pass
    x, y = res1.to_dataframe("out").values.flatten(), np.asarray(res3).flatten()
    assert np.mean(np.abs(x - y)) == pytest.approx(0, rel=accuracy, abs=accuracy)
    x, y = res2.to_dataframe(("rnn", "li_op/v")).values.flatten(), np.asarray(res4).flatten()
    assert np.mean(np.abs(x - y)) == pytest.approx(0, rel=accuracy, abs=accuracy)


def test_3_7_optimization():
    """Tests optimization functions of Network class.
    """

    # preparations
    ##############

    # rnn parameters
    dt = 1e-2
    n = 10
    n_out = 3
    steps = 100
    W = np.random.randn(n, n)
    W_out = np.random.randn(n_out, n)
    x = np.random.randn(steps, n)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # forward simulation of target dynamics
    net = Network(dt, device="cpu")
    net.add_diffeq_node("rnn", node, input_var=in_var, output_var=out_var, weights=W, source_var=s_var,
                        target_var=t_var)
    net.add_func_node("output", n_out, "identity")
    net.add_edge("rnn", "output", weights=W_out, train=None)
    y0 = net.state
    obs = net.run(x, sampling_steps=1, enable_grad=False, verbose=False)
    target = obs.to_numpy("out")

    # fit readout weights via different optimization mechanisms
    ###########################################################

    n_epochs = 1000
    input_epochs = [x for _ in range(n_epochs)]
    target_epochs = [target for _ in range(n_epochs)]

    # fit readout weights via bptt
    net.pop_edge("rnn", "output")
    net.add_edge("rnn", "output", weights=np.random.randn(n_out, n), train="gd")
    net.reset(y0)
    net.fit_bptt(input_epochs, target_epochs, sampling_steps=1, verbose=False, lr=0.1, optimizer="adam")
    bptt_weights = net.get_edge("rnn", "output").weights.detach().numpy()

    # fit readout weights via rls
    net.pop_edge("rnn", "output")
    net.add_edge("rnn", "output", train="rls", beta=0.99, alpha=1.0)
    net.reset(y0)
    net.fit_rls(input_epochs, target_epochs, update_steps=1, verbose=False)
    rls_weights = net.get_edge("rnn", "output").weights.detach().numpy()

    # fit readout weights via ridge regression
    net.reset(y0)
    net.pop_node("output")
    obs_ridge = net.fit_ridge(x, target, sampling_steps=1, add_readout_node=False, verbose=False, alpha=0.0)
    ridge_weights = obs_ridge.to_numpy("w_out").T

    # testing
    #########

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(ncols=2)
    # ax = axes[0]
    # ax.imshow(W_out, aspect="auto")
    # ax.set_title("target")
    # ax = axes[1]
    # ax.imshow(np.round(rls_weights, decimals=2), aspect="auto")
    # ax.set_title("fit")
    # plt.show()

    # test bppt fit
    assert np.mean((W_out - bptt_weights)**2) == pytest.approx(0.0, 0.5, 0.5)

    # test rls fit
    assert np.mean((W_out - rls_weights) ** 2) == pytest.approx(0.0, 0.5, 0.5)

    # test ridge fit
    assert np.mean((W_out - ridge_weights) ** 2) == pytest.approx(0.0, 0.5, 0.5)
