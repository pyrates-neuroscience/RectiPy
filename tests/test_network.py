"""Test suite for the network functionalities, which serves as the main user interface.
"""

# imports
from rectipy.rnn_layer import RNNLayer, SRNNLayer
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
    node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh_pop"
    node_spiking = "neuron_model_templates.spiking_neurons.qif.qif_pop"
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
    net2 = Network(rnn)
    net3 = Network.from_yaml(node, weights=weights, input_var="I_ext", output_var=out_var, source_var=s_var,
                             target_var="r_in", clear=True, verbose=False, op="li_op")
    net4 = Network.from_yaml(node, weights=weights, input_var=in_var, output_var=out_var, source_var=s_var,
                             target_var=t_var, clear=True, verbose=False, train_params=["weight"],
                             record_vars=["li_op/u"])
    net5 = Network.from_yaml(node_spiking, weights=weights, input_var="I_ext", output_var="s", source_var="s",
                             target_var="s_in", op="qif_op", spike_var="spike", spike_def="v",  clear=True,
                             verbose=False)

    # these tests should pass
    assert isinstance(net1.rnn_layer, RNNLayer)
    assert isinstance(net5.rnn_layer, SRNNLayer)
    assert net2.rnn_layer == rnn
    assert len(net1._var_map) == 0
    assert len(net3._var_map) == 2
    assert len(net1.rnn_layer.train_params) == 0
    assert len(net4.rnn_layer.train_params) == 1
    assert list(net4.rnn_layer.record(["li_op/u"]))

    # these tests should fail
    with pytest.raises(FileNotFoundError):
        RNNLayer.from_yaml("neuron_model_templates.rate_neurons.freaky_integrator.tanh_pop", weights=weights,
                           source_var=s_var, target_var=t_var, input_var=in_var, output_var=out_var, clear=True,
                           verbose=False)
    with pytest.raises(AttributeError):
        RNNLayer.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tan_pop", weights=weights,
                           source_var=s_var, target_var=t_var, input_var=in_var, output_var=out_var, clear=True,
                           verbose=False)
    with pytest.raises(KeyError):
        RNNLayer.from_yaml(node, weights=weights, source_var="x", target_var=t_var, input_var=in_var,
                           output_var=out_var, clear=True, verbose=False)


def test_4_2_input_layer():
    """Tests input layer properties of Network class.
    """
    pass


def test_4_3_output_layer():
    """Tests output layer properties of Network class.
    """
    pass


def test_4_4_compile():
    """Tests compile functionalities of Network class.
    """
    pass


def test_4_5_forward():
    """Tests forward method of Network class.
    """
    pass


def test_4_6_parameters():
    """Tests parameters method of Network class.
    """
    pass


def test_4_7_simulation():
    """Tests simulation functionalities of Network class.
    """

    pass


def test_4_8_optimization():
    """Tests optimization functions of Network class.
    """

    pass
