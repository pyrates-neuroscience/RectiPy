"""Test suite for the network functionalities, which serves as the main user interface.
"""

# imports
from rectipy.rnn_layer import RNNLayer, SRNNLayer
from rectipy.input_layer import LinearStatic, Linear
from rectipy import Network
import torch
import pytest
import numpy as np
from pyrates import clear_frontend_caches


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


def test_4_1_init():
    """Tests initialization of Network class.
    """

    clear_frontend_caches()

    # parameters
    n = 10
    weights = np.random.randn(n, n)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    node_spiking = "neuron_model_templates.spiking_neurons.qif.qif"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # rnn layer initialization
    rnn = RNNLayer.from_yaml(node, weights=weights, source_var=s_var, target_var=t_var, input_var=in_var,
                             output_var=out_var, clear=True, verbose=False)

    # different network initializations
    net1 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False)
    net2 = Network(n, rnn)
    net3 = Network.from_yaml(node, weights=weights, input_var="I_ext", output_var=out_var, source_var=s_var,
                             target_var="r_in", clear=True, verbose=False, op="li_op")
    net4 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, train_params=["weights"])
    net5 = Network.from_yaml(node_spiking, weights=weights, input_var="I_ext", output_var="s", source_var="s",
                             target_var="s_in", op="qif_op", spike_var="spike", spike_def="v",  clear=True,
                             verbose=False, dtype=torch.float32)

    # these tests should pass
    assert isinstance(net1.rnn_layer, RNNLayer)
    assert isinstance(net5.rnn_layer, SRNNLayer)
    assert isinstance(net1[0], RNNLayer)
    assert net2.rnn_layer == rnn
    assert len(net3._var_map) - len(net1._var_map) == 2
    assert len(net1.rnn_layer.train_params) == 0
    assert len(net4.rnn_layer.train_params) == 1
    assert net4["v"].shape[0] == n
    assert net1.rnn_layer.y.dtype == torch.float64
    assert net5.rnn_layer.y.dtype == torch.float32

    # these tests should fail
    with pytest.raises(FileNotFoundError):
        RNNLayer.from_yaml("neuron_model_templates.rate_neurons.freaky_integrator.tanh", weights=weights,
                           source_var=s_var, target_var=t_var, input_var=in_var, output_var=out_var, clear=True,
                           verbose=False)
    with pytest.raises(AttributeError):
        RNNLayer.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tan", weights=weights,
                           source_var=s_var, target_var=t_var, input_var=in_var, output_var=out_var, clear=True,
                           verbose=False)
    with pytest.raises(KeyError):
        RNNLayer.from_yaml(node, weights=weights, source_var="x", target_var=t_var, input_var=in_var,
                           output_var=out_var, clear=True, verbose=False)


def test_4_2_input_layer():
    """Tests input layer properties of Network class.
    """

    # rnn parameters
    n = 10
    weights = np.random.randn(n, n)
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh"
    in_var = "li_op/I_ext"
    out_var = "tanh_op/r"
    s_var = "tanh_op/r"
    t_var = "li_op/r_in"

    # input parameters
    m = 3
    x = torch.randn(m, dtype=torch.float32)

    # different network initializations
    net1 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net1", dtype=torch.float32)
    net2 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net2", dtype=torch.float32)
    net3 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net3", dtype=torch.float32)
    net4 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net4", dtype=torch.float32)

    # add input layer
    net1.add_input_layer(m, trainable=False, dtype=torch.float32)
    net2.add_input_layer(m, weights=np.random.randn(m, n), trainable=False, dtype=torch.float32)
    net3.add_input_layer(m, trainable=True, dtype=torch.float32)
    net4.add_input_layer(m, dtype=torch.float64)
    net1.compile()
    net2.compile()
    net3.compile()
    net4.compile()

    # these tests should pass
    assert isinstance(net1.input_layer, LinearStatic)
    assert isinstance(net3.input_layer, Linear)
    assert isinstance(net1[0], LinearStatic)
    assert tuple(net2.input_layer.weight.shape) == (n, m)
    assert net4.input_layer.weight.dtype == torch.float64
    assert tuple(net1.forward(x).shape) == (n,)
    net1.remove_input_layer()
    net1.compile()
    assert isinstance(net1[0], RNNLayer)

    # these tests should fail
    with pytest.raises(RuntimeError):
        net4.forward(x)
        net1.forward(x)
        net2.forward(torch.randn(m+1))
    with pytest.raises(ValueError):
        net1.add_input_layer(m, weights=np.random.randn(m+1, n+1))


def test_4_3_output_layer():
    """Tests output layer properties of Network class.
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
    k = 3
    out_weights = np.random.randn(n, k)

    # input definition
    x = torch.randn(n, dtype=torch.float64)

    # different network initializations
    net1 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net1", dtype=torch.float64)
    net2 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net2", dtype=torch.float64)
    net3 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net3", dtype=torch.float64)
    net4 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net4", dtype=torch.float64)
    net5 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net5", dtype=torch.float32)

    # add output layers
    net1.add_output_layer(k, weights=out_weights)
    net2.add_output_layer(k, weights=out_weights, activation_function='sigmoid')
    net3.add_output_layer(k, trainable=True)
    net4.add_output_layer(k, trainable=True, bias=False)
    net5.add_output_layer(k, dtype=torch.float32)
    net1.compile()
    net2.compile()
    net5.compile()

    # these tests should pass
    assert isinstance(net1.output_layer, torch.nn.Sequential)
    assert isinstance(net1.output_layer[0], LinearStatic)
    assert isinstance(net3.output_layer[0], Linear)
    assert isinstance(net1.output_layer[1], torch.nn.Identity)
    assert isinstance(net2.output_layer[1], torch.nn.Sigmoid)
    assert len(list(net1.parameters())) == 0
    assert len(list(net3.parameters())) == 2
    assert len(list(net4.parameters())) == 1
    assert net5.output_layer[0].weight.dtype == torch.float32
    assert tuple(net1.forward(x).shape) == (k,)
    assert np.mean(np.abs(net1.forward(x).detach().numpy() - net2.forward(x).detach().numpy())) > 0.0
    net1.remove_output_layer()
    net1.compile()
    assert tuple(net1.forward(x).shape) == (n,)

    # these tests should fail
    with pytest.raises(RuntimeError):
        net5.forward(x)


def test_4_4_compile():
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
    k = 3

    # input parameters
    m = 3
    x = torch.randn(m, dtype=torch.float64)

    # network initialization
    net = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                            target_var=t_var, clear=True, verbose=False, file_name="net1", dtype=torch.float64)

    # these tests should pass
    net.compile()
    assert isinstance(net.model, torch.nn.Sequential)
    assert len(net) == 1
    net.add_input_layer(m, dtype=torch.float64)
    net.compile()
    assert len(net) == 2
    y1 = net.forward(x)
    net.add_output_layer(k, dtype=torch.float64)
    net.compile()
    y2 = net.forward(x)
    assert len(net) == 3
    assert y2.shape[0] - y1.shape[0] == k-n

    # these tests should fail
    net.remove_input_layer()
    net.compile()
    with pytest.raises(RuntimeError):
        net.forward(x)


def test_4_5_parameters():
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
    net1 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net1", dtype=torch.float64)
    net2 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net2", dtype=torch.float64,
                             train_params=['weights', 'li_op/tau'])

    # test number of parameters
    assert len(list(net1.parameters())) == 0
    assert len(list(net2.parameters())) == 2

    # add input layers
    net1.add_input_layer(m, trainable=True)
    net2.add_input_layer(m, trainable=False)
    net1.compile()
    net2.compile()

    # test number of parameters
    assert len(list(net1.parameters())) == 1
    assert len(list(net2.parameters())) == 2

    # add output layers
    net1.add_output_layer(k, trainable=True, bias=False)
    net2.add_output_layer(k, trainable=True, bias=True)
    net1.compile()
    net2.compile()

    # test number of parameters
    assert len(list(net1.parameters())) == 2
    assert len(list(net2.parameters())) == 4


def test_4_6_simulation():
    """Tests simulation functionalities of Network class.
    """

    # rnn parameters
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
    net1 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net1", dtype=torch.float64)
    net2 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net2", dtype=torch.float64,
                             record_vars=['li_op/v'])
    net3 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, file_name="net2", dtype=torch.float64,
                             record_vars=['li_op/v']
                             )
    net3.compile()

    # run simulations
    res1 = net1.run(inputs=x, sampling_steps=2, verbose=False)
    res2 = net2.run(inputs=x, record_output=False, record_vars=[('li_op/v', False)], verbose=False)
    res3, res4 = [], []
    for step in range(steps):
        out = net3.forward(x[step, :])
        if step % 2 == 0:
            res3.append(out.detach().numpy())
        res4.append(net3['li_op/v'].detach().numpy())

    # these tests should pass
    for r1, r2 in zip(res1['out'], res3):
        assert np.mean(np.abs(r1 - r2)) == pytest.approx(0, rel=accuracy, abs=accuracy)
    for r1, r2 in zip(res2['li_op/v'], res4):
        assert np.mean(np.abs(r1 - r2)) == pytest.approx(0, rel=accuracy, abs=accuracy)


def test_4_7_optimization():
    """Tests optimization functions of Network class.
    """

    pass
