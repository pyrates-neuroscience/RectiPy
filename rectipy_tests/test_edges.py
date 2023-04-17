"""Test suite for the  feedforward layer functionalities.
"""

# imports
from rectipy.edges import Linear, RLS
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

def test_1_1_linear():
    """Testing functionalities of the `rectipy.edges.Linear` class.
    """

    # preparations
    ##############

    # parameters
    dtype = torch.float64
    n = 10
    m = 2
    w1 = np.random.randn(n, m)
    x = torch.randn(n, dtype=dtype)

    # create torch comparison layer
    in_torch = TorchLinear(in_features=n, out_features=m, bias=False, dtype=dtype)

    # create different instances of the input layer
    lin1 = Linear(n, m)
    lin2 = Linear(m, n)
    lin3 = Linear(n, m, weights=w1.T + 2.0)
    lin4 = Linear(n, m, weights=w1, dtype=torch.float32)
    lin5 = Linear(n, m, weights=w1, detach=False)
    lin6 = Linear(n, m, weights=in_torch.weight)

    # these tests should pass
    #########################

    # test successful initialization
    assert isinstance(lin1, Linear)

    # test correct handling of layer weights
    assert lin2.weights.shape == w1.shape
    assert lin1.weights.shape[0] == lin2.weights.shape[1]
    assert torch.sum(lin5.weights.detach() - w1.T).numpy() == pytest.approx(0.0, rel=accuracy, abs=accuracy)
    assert lin3.weights.dtype == torch.float64
    assert lin4.weights.dtype == torch.float32

    # test correct handling of trainable parameters (the weights)
    assert len(list(lin5.parameters())) - len(list(lin4.parameters())) == 1
    assert len(list(lin1.parameters())) == 0
    assert lin1.weights.requires_grad is False
    assert lin5.weights.requires_grad is True

    # test correctness of forward function
    assert np.abs(torch.sum(lin5.forward(x) - lin3.forward(x)).detach().numpy()) > 0.0
    assert np.abs(torch.sum(in_torch.forward(x) - lin6.forward(x)).detach().numpy()) == pytest.approx(0.0, rel=accuracy, abs=accuracy)

    # these tests should fail
    #########################

    # test whether correct errors are thrown for erroneous initializations
    with pytest.raises(ValueError):
        Linear(n, m, weights=np.random.randn(n+1, m+1))

    # test whether correct errors are thrown for wrong usage of the forward method
    with pytest.raises(RuntimeError):
        lin4.forward(x)
        lin1.forward(torch.randn(m))


def test_1_2_rls_layer():
    """Tests the functionalities of the `rectipy.edges.RLSLayer` class.
    """

    # preparations
    ##############

    # parameters
    dtype = torch.float64
    n = 10
    m = 2
    w1 = np.random.randn(n, m)
    x = torch.randn(n, dtype=dtype)
    y = torch.randn(m, dtype=dtype)

    # create different instances of the input layer
    rls1 = RLS(n, m)
    rls2 = RLS(n, m, weights=w1)
    rls3 = RLS(n, m, weights=w1, beta=0.5)
    rls4 = RLS(n, m, weights=w1, alpha=0.5)

    # these tests should pass
    #########################

    # test correct initialization
    assert isinstance(rls1, RLS)
    assert torch.sum(rls2.weights - w1.T).numpy() == pytest.approx(0.0, rel=accuracy, abs=accuracy)
    assert rls1.P.shape[0] == n
    assert len(list(rls2.parameters())) == 0

    # calculate different forward passes of the RLS layer for preparation
    r1_1 = rls1.forward(x)
    r1_2 = rls1.forward(x)
    for rls in [rls2, rls3, rls4]:
        y_hat = rls.forward(x)
        rls.update(x, y_hat, y)
    r2 = rls2.forward(x)
    r3 = rls3.forward(x)
    r4 = rls4.forward(x)

    # test correctness of forward method
    assert r1_1.shape[0] == m
    assert np.abs(torch.sum(r1_1 - r1_2).detach().numpy()) == pytest.approx(0.0, rel=accuracy, abs=accuracy)
    assert np.abs(torch.sum(r2 - r3).detach().numpy()) > 0
    assert np.abs(torch.sum(r3 - r4).detach().numpy()) > 0

    # these tests should fail
    #########################

    # initialization
    with pytest.raises(ValueError):
        RLS(n, m, alpha=-0.5)
        RLS(n, m, beta=1.5)
        RLS(n + 1, m, weights=w1)

    # forwarding
    with pytest.raises(RuntimeError):
        rls1.forward(torch.randn(n+1, dtype=dtype))
        rls1.forward(torch.randn(n, dtype=torch.float32))
        rls1.update(x, torch.randn(m+1, dtype=dtype), torch.randn(m+1, dtype=dtype))
