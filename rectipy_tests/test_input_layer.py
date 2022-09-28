"""Test suite for the input layer functionalities.
"""

# imports
from rectipy.input_layer import InputLayer, LinearStatic, Linear
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
    print("============================")
    print("| Test Suite : Input layer |")
    print("============================")


# test accuracy
accuracy = 1e-4


# tests
#######


def test_1_1_linear_static():
    """Testing functionalities of the `rectipy.input_layer.LinearStatic` class.
    """

    # parameters
    n = 10
    weights = torch.randn(n, n)
    x = torch.randn(n)

    # initialize static input layer
    inp = LinearStatic(weights=weights)

    # these tests should pass
    assert isinstance(inp, LinearStatic)
    assert inp.weight.shape == weights.shape
    assert torch.sum(inp.forward(x)-weights @ x).numpy() == pytest.approx(0.0, rel=accuracy, abs=accuracy)

    # these tests should fail
    with pytest.raises(TypeError):
        LinearStatic(np.random.randn(n, n))
    with pytest.raises(RuntimeError):
        inp.forward(torch.randn(n+1))


def test_1_2_input_layer():
    """Testing functionalities of the `rectipy.input_layer.InputLayer` class.
    """

    # parameters
    n = 10
    m = 2
    weights = np.random.randn(n, m)
    x = torch.randn(m, dtype=torch.float64)

    # create different instances of the input layer
    in1 = InputLayer(n, m)
    in2 = InputLayer(m, n)
    in3 = InputLayer(n, m, weights=weights.T)
    in4 = InputLayer(n, m, weights=weights, dtype=torch.float32)
    in5 = InputLayer(n, m, weights=weights, trainable=True)

    # these tests should pass
    assert isinstance(in1, LinearStatic)
    assert isinstance(in5, Linear)
    assert in1.weight.shape[0] == in2.weight.shape[1]
    assert torch.sum(in3.weight - weights).numpy() == pytest.approx(0.0, rel=accuracy, abs=accuracy)
    assert in3.weight.dtype == torch.float64
    assert in4.weight.dtype == torch.float32
    assert np.abs(torch.sum(in5.forward(x) - in3.forward(x)).detach().numpy()) > 0.0
    assert len(list(in5.parameters())) - len(list(in4.parameters())) == 1

    # these tests should fail
    with pytest.raises(ValueError):
        InputLayer(n, m, weights=np.random.randn(n+1, m+1))
    with pytest.raises(RuntimeError):
        in4.forward(x)
        in5.forward(torch.randn(n))
