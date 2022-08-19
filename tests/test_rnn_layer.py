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
def rate(t, y, I_ext, weights, tau):
    return -y/tau + I_ext + weights @ torch.tanh(y)


# tests
#######


def test_3_1_rnn_init():
    """Tests initialization options of the rnn layer.
    """

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    args = (torch.zeros((n,)), torch.zeros((n,)), torch.tensor(weights), 1.0)

    # create different instances of RNNLayer
    rnn1 = RNNLayer(func, args, 1, list(range(n)))
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


def test_3_2_detach():
    """Tests the detach function of the RNN layer.
    """

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    args = (torch.zeros((n,), requires_grad=True), torch.zeros((n,), requires_grad=True),
            torch.tensor(weights, requires_grad=True), 1.0)
    args2 = tuple([a.clone() if type(a) is torch.Tensor else a for a in args])

    # create different instances of RNNLayer
    rnn1 = RNNLayer(func, args, 1, list(range(n)))
    rnn2 = RNNLayer(func, args2, 1, list(range(n)))
    rnn2.detach()

    # these tests should pass
    assert rnn1.y.requires_grad
    assert not rnn2.y.requires_grad
    for arg1, arg2 in zip(rnn1.args, rnn2.args):
        if type(arg1) is torch.Tensor:
            assert arg1.requires_grad
            assert not arg2.requires_grad

def test_3_3_forward():
    """Tests forward function of the RNN layer.
    """

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    dtype = torch.float64
    args = (torch.zeros((n,), dtype=dtype), torch.zeros((n,), dtype=dtype), torch.tensor(weights, dtype=dtype), 1.0)
    inp_idx = 1
    inp = torch.randn(n, dtype=dtype)

    # create different instances of RNNLayer
    rnn1 = RNNLayer(func, args, input_ext=inp_idx, output=list(range(n)))
    rnn2 = RNNLayer.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh_node", weights=weights,
                              source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                              output_var="tanh_op/r", clear=True, float_precision="float64")
    rnn3 = RNNLayer(func, args + (1.0, ), input_ext=inp_idx, output=list(range(n)))
    rnn4 = RNNLayer(func, args, input_ext=inp_idx+2, output=list(range(n)))
    rnn5 = RNNLayer(func, args, input_ext=inp_idx, output=[0, 1, 2])

    # detach the rnns
    for rnn in [rnn1, rnn2, rnn3, rnn4, rnn5]:
        rnn.detach()

    # calculate layer outputs
    steps = 10
    out1 = [rnn1.forward(inp).numpy() for _ in range(steps)]
    out2 = [rnn2.forward(inp).numpy() for _ in range(steps)]
    out4 = rnn4.forward(inp).numpy()
    out5 = rnn5.forward(inp).numpy()

    # these tests should pass
    assert out1[0].shape[0] == n
    for o1, o2 in zip(out1, out2):
        assert np.mean(np.abs(o1-o2)) == pytest.approx(0, rel=accuracy, abs=accuracy)
    assert np.mean(np.abs(out4 - out1[0])) > 0
    assert out5.shape[0] == 3

    # these tests should fail
    with pytest.raises(RuntimeError):
        rnn2.forward(torch.randn(n+1))
        rnn3.forward(inp)
        rnn1.forward(np.random.randn(n))
