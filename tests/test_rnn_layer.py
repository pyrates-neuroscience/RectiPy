"""Test suite for the input layer functionalities.
"""

# imports
from rectipy.rnn_layer import RNNLayer, SRNNLayer
import torch
import pytest
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


# Utility
#########


def setup_module():
    print("\n")
    print("============================")
    print("| Test Suite : Input layer |")
    print("============================")


# test accuracy
accuracy = 1e-4


# rate network function
def rate(t, y, dy, I_ext, r_in, weights, tau):
    dy[:] = -y/tau + I_ext + weights @ torch.tanh(r_in)
    return dy


# tests
#######


def test_3_1_rnn_init():
    """Tests initialization options of the rnn layer.
    """

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    args = (torch.zeros((n,)), torch.zeros((n,)), torch.zeros((n,)), torch.zeros((n,)), torch.tensor(weights), 1.0)

    # create different instances of RNNLayer
    rnn1 = RNNLayer(func, args, 3, list(range(n)))
    rnn2 = RNNLayer.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh_node", weights=weights,
                              source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                              output_var="tanh_op/r", clear=True)
    rnn3 = SRNNLayer.from_yaml("neuron_model_templates.spiking_neurons.qif.qif_pop", weights=weights,
                               source_var="qif_op/s", target_var="qif_op/s_in", input_var_ext="qif_op/I_ext",
                               output_var="qif_op/s", spike_var="qif_op/v", input_var_net="qif_op/spike",
                               spike_threshold=1e3, spike_reset=-1e3, clear=True)
    rnn4 = RNNLayer.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh_node", weights=weights,
                              source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                              output_var="tanh_op/r", clear=True, train_params=["weight"], record_vars=["li_op/u"])

    # these tests should pass
    assert isinstance(rnn1, RNNLayer)
    assert isinstance(rnn2, RNNLayer)
    assert isinstance(rnn3, SRNNLayer)
    assert len(rnn2.y) == n
    assert len(rnn3.dy) == 2*n
    assert len(list(rnn4.parameters())) - len(list(rnn2.parameters())) == 1
    assert list(rnn4.record(['li_op/u']))[0].shape[0] == n

    # these tests should fail
    with pytest.raises(KeyError):
        list(rnn2.record(['li_op/u']))
