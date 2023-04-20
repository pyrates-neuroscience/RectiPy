"""Test suite for the rnn layer functionalities.
"""

# imports
from rectipy.nodes import RateNet, SpikeNet
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
    print("==========================")
    print("| Test Suite : RNN Layer |")
    print("==========================")


# test accuracy
accuracy = 1e-3


# rate network function
def rate(t, y, I_ext, weights, tau):
    return -y/tau + I_ext + weights @ torch.tanh(y)


# tests
#######


def test_2_1_ratenet_init():
    """Tests initialization options of the rnn layer.
    """

    clear_frontend_caches()

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    args = (torch.zeros((n,)), torch.zeros((n,)), torch.tensor(weights), 1.0)

    # create different instances of RNNLayer
    rnn1 = RateNet(func, args, {"out": [0, n]}, {"in": 0})
    rnn2 = RateNet.from_pyrates("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=weights,
                                source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                                output_var="tanh_op/r", clear=True, verbose=False)
    rnn3 = SpikeNet.from_pyrates("neuron_model_templates.spiking_neurons.qif.qif", weights=weights,
                                 source_var="qif_op/s", target_var="qif_op/s_in", input_var="qif_op/I_ext",
                                 output_var="qif_op/s", spike_def="qif_op/v", spike_var="qif_op/spike",
                                 spike_threshold=1e3, spike_reset=-1e3, clear=True, verbose=False, dtype=torch.float32)
    rnn4 = RateNet.from_pyrates("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=weights,
                                source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                                output_var="tanh_op/r", clear=True, train_params=["weights"], verbose=False)

    # these tests should pass
    assert isinstance(rnn1, RateNet)
    assert isinstance(rnn2, RateNet)
    assert isinstance(rnn3, SpikeNet)
    assert len(rnn2.y) == n
    assert len(rnn3.y) == 2*n
    assert len(list(rnn4.parameters())) - len(list(rnn2.parameters())) == 1
    assert rnn4['li_op/v'].shape[0] == n
    assert rnn3.y.dtype == torch.float32
    assert rnn4.y.dtype == torch.float64

    # these tests should fail
    with pytest.raises(KeyError):
        _ = rnn2['li_op/u']


def test_2_2_detach():
    """Tests the detach function of the RNN layer.
    """

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    args = (torch.zeros((n,), requires_grad=False), torch.zeros((n,), requires_grad=False),
            torch.tensor(weights, requires_grad=False), 1.0)
    args2 = tuple([a.clone() if type(a) is torch.Tensor else a for a in args])

    # create different instances of RNNLayer
    rnn1 = RateNet(func, args, {"out": [0, n]}, {"in": 0, "weights": 1}, train_params=["weights"])
    rnn2 = RateNet(func, args2, {"out": [0, n]}, {"in": 0, "weights": 1}, train_params=["weights"])
    rnn2.detach()

    # these tests should pass
    assert rnn1.y.requires_grad
    assert not rnn2.y.requires_grad
    assert rnn1["weights"].requires_grad
    assert not rnn2["weights"].requires_grad


def test_2_3_forward():
    """Tests forward function of the RNN layer.
    """

    clear_frontend_caches()

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    dtype = torch.float64
    args = (torch.zeros((n,), dtype=dtype), torch.zeros((n,), dtype=dtype), torch.tensor(weights, dtype=dtype), 1.0)
    inp = torch.randn(n, dtype=dtype)

    # create different instances of RNNLayer
    rnn1 = RateNet(func, args, {"out": [0, n]}, {"in": 0})
    rnn2 = RateNet.from_pyrates("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=weights,
                                source_var="tanh_op/r", target_var="li_op/r_in", input_var="li_op/I_ext",
                                output_var="tanh_op/r", clear=True, verbose=False)
    rnn3 = RateNet(func, args + (1.0,), {"out": [0, n]}, {"in": 0})
    rnn4 = RateNet(func, args, {"out": [0, n]}, {"in": 2})
    rnn5 = RateNet(func, args, {"out": [0, 3]}, {"in": 0})

    # detach the rnns
    for rnn in [rnn1, rnn2, rnn3, rnn4, rnn5]:
        rnn.detach()

    # calculate layer outputs
    steps = 10
    out1 = [rnn1.forward(inp).numpy() for _ in range(steps)]
    out2 = [rnn2.forward(inp).numpy() for _ in range(steps)]
    out4 = [rnn4.forward(inp).numpy() for _ in range(steps)]
    out5 = rnn5.forward(inp).numpy()

    # these tests should pass
    assert out1[0].shape[0] == n
    for o1, o2 in zip(out1, out2):
        assert np.mean(np.abs(o1-o2)) == pytest.approx(0, rel=accuracy, abs=accuracy)
    assert np.mean(np.abs(out4[-1] - out1[-1])) > 0
    assert out5.shape[0] == 3

    # these tests should fail
    with pytest.raises(RuntimeError):
        rnn2.forward(torch.randn(n+1))
        rnn3.forward(inp)
        rnn1.forward(np.random.randn(n))


def test_2_4_reset():
    """Tests reset method of RNNLayer
    """

    # parameters
    func = rate
    n = 10
    weights = np.random.randn(n, n)
    y0 = np.random.randn(n)
    args = (torch.tensor(y0), torch.zeros((n,)), torch.tensor(weights), 1.0)
    y1 = y0[:] + 1.0
    x = torch.randn(n)

    # create instance of RNNLayer
    rnn = RateNet(func, args, {"out": [0, n]}, {"in": 0})

    # collect states
    r1 = rnn.forward(x)
    r2 = rnn.forward(x)
    rnn.reset(y0)
    r3 = rnn.forward(x)
    rnn.reset(y0[0:3], idx=np.arange(0, 3))
    r5 = rnn.forward(x)
    rnn.reset(y1)
    r4 = rnn.forward(x)

    # these tests should pass
    for z1, z2 in [(r1, r2), (r1, r4), (r1, r5)]:
        assert np.mean(np.abs(z1.detach().numpy() - z2.detach().numpy())) > 0
    assert np.mean(r1.detach().numpy() - r3.detach().numpy()) == pytest.approx(0, abs=accuracy, rel=accuracy)
    assert np.mean(r1.detach().numpy()[0:3] - r5.detach().numpy()[0:3]) == pytest.approx(0, abs=accuracy, rel=accuracy)
    assert np.mean(r2.detach().numpy()[3:n] - r5.detach().numpy()[3:n]) == pytest.approx(0, abs=accuracy, rel=accuracy)

    # these tests should fail
    with pytest.raises(RuntimeError):
        rnn.reset(np.random.randn(n+1))
        rnn.reset(y0, idx=np.arange(0, n+1))
