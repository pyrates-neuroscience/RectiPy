"""Test suite for the output layer functionalities.
"""

# imports
from rectipy.output_layer import OutputLayer
from rectipy.input_layer import LinearStatic, Linear
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
    print("=============================")
    print("| Test Suite : Output layer |")
    print("=============================")


# test accuracy
accuracy = 1e-4


# tests
#######


def test_2_1_output_layer():
    """Tests the functionalities of the `rectipy.output_layer.OutputLayer` class.
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
    out4 = OutputLayer(n, m, weights=weights, trainable=True)
    out5 = OutputLayer(n, m, weights=weights, activation_function='tanh')
    out6 = OutputLayer(n, m, dtype=torch.float32)
    out7 = OutputLayer(n, m, trainable=True, bias=False)

    # these tests should pass
    assert isinstance(out1, torch.nn.Sequential)
    assert isinstance(out1[0], LinearStatic)
    assert isinstance(out4[0], Linear)
    assert isinstance(out1[1], torch.nn.Identity)
    assert isinstance(out5[1], torch.nn.Tanh)
    assert out1[0].weight.shape[0] == out2[0].weight.shape[1]
    assert torch.sum(out3[0].weight - weights).numpy() == pytest.approx(0.0, rel=accuracy, abs=accuracy)
    assert out3[0].weight.dtype == torch.float64
    assert out6[0].weight.dtype == torch.float32
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
