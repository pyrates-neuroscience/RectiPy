"""Test suite for the  feedforward layer functionalities.
"""

# imports
from rectipy.ffwd_layer import GradientDescentLayer, Linear, RLSLayer, LayerStack
from torch.nn import Linear as TorchLinear
import pytest
import numpy as np
import torch

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


# Utility
#########


def setup_module():
    print("\n")
    print("===================================")
    print("| Test Suite : Feed-Forward layer |")
    print("===================================")


# test accuracy
accuracy = 1e-4


# tests
#######

def test_1_1_linear_layer():
    """Testing functionalities of the `rectipy.ffwd_layer.Linear` class.
    """

    # preparations
    ##############

    # parameters
    dtype = torch.float64
    n = 10
    m = 2
    w1 = np.random.randn(n, m)
    x = torch.randn(m, dtype=torch.float64)

    # create torch comparison layer
    in_torch = TorchLinear(in_features=n, out_features=m, bias=False)

    # create different instances of the input layer
    in1 = Linear(n, m)
    in2 = Linear(m, n)
    in3 = Linear(n, m, weights=w1.T)
    in4 = Linear(n, m, weights=w1, dtype=torch.float32)
    in5 = Linear(n, m, weights=w1, detach=False)
    in6 = Linear(n, m, weights=in_torch.weight)

    # these tests should pass
    #########################

    # test successful initialization
    assert isinstance(in1, Linear)

    # test correct handling of layer weights
    assert in1.weights.shape == w1.shape
    assert in1.weights.shape[0] == in2.weights.shape[1]
    assert torch.sum(in3.weights - w1).numpy() == pytest.approx(0.0, rel=accuracy, abs=accuracy)
    assert in3.weights.dtype == torch.float64
    assert in4.weights.dtype == torch.float32

    # test correct handling of trainable parameters (the weights)
    assert len(list(in5.parameters())) - len(list(in4.parameters())) == 1
    assert len(list(in1.parameters())) == 0
    assert in1.weights.requires_grad is False
    assert in5.weights.requires_grad is False

    # test correctness of forward function
    assert np.abs(torch.sum(in5.forward(x) - in3.forward(x)).detach().numpy()) > 0.0
    assert np.abs(torch.sum(in_torch.forward(x) - in6.forward(x)).detach().numpy()) == pytest.approx(0.0, rel=accuracy, abs=accuracy)

    # these tests should fail
    #########################

    # test whether correct errors are thrown for erroneous initializations
    with pytest.raises(ValueError):
        Linear(n, m, weights=np.random.randn(n+1, m+1))

    # test whether correct errors are thrown for wrong usage of the forward method
    with pytest.raises(RuntimeError):
        in4.forward(x)
        in1.forward(torch.randn(n))


def test_1_3_layerstack():
    """Tests the functionalities of the `rectipy.ffwd_layer.LayerStack` class.
    """

    # parameters
    n = 10
    m = 2
    weights = np.random.randn(m, n)
    x = torch.randn(n, dtype=torch.float64)

    # create different output layer instances
    out1 = OutputLayer(n, m)
    out2 = OutputLayer(m, n)
    out3 = OutputLayer(n, m, weights=weights)
    out4 = OutputLayer(n, m, weights=weights, train=True)
    out5 = OutputLayer(n, m, weights=weights, activation_function='tanh')
    out6 = OutputLayer(n, m, dtype=torch.float32)
    out7 = OutputLayer(n, m, train=True, bias=False)

    # these tests should pass
    assert isinstance(out1, torch.nn.Sequential)
    assert isinstance(out1[0], LinearStatic)
    assert isinstance(out4[0], Linear)
    assert isinstance(out1[1], torch.nn.Identity)
    assert isinstance(out5[1], torch.nn.Tanh)
    assert out1[0].weights.shape[0] == out2[0].weights.shape[1]
    assert torch.sum(out3[0].weights - weights).numpy() == pytest.approx(0.0, rel=accuracy, abs=accuracy)
    assert out3[0].weights.dtype == torch.float64
    assert out6[0].weights.dtype == torch.float32
    assert np.abs(torch.sum(out4.forward(x) - out3.forward(x)).detach().numpy()) > 0.0
    assert np.abs(torch.sum(out5.forward(x) - out3.forward(x)).detach().numpy()) > 0.0
    assert len(list(out4.parameters())) - len(list(out3.parameters())) == 2
    assert len(list(out7.parameters())) == 1
    assert len(out5) == 2

    # these tests should fail
    with pytest.raises(ValueError):
        OutputLayer(n, m, weights=np.random.randn(n+1, m))
    with pytest.raises(RuntimeError):
        out6.forward(x)
        out2.forward(x)
    with pytest.raises(ValueError):
        OutputLayer(n, m, activation_function='invalid')
