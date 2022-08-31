# Changelog

## 0.5

### 0.5.0

- reduced overhead of `InputLayer` and `OutputLayer` by making them return instances of `torch.nn.Linear` or `rectipy.input_layer.LinearStatic` upon initialization
- reduced overhead of `Network.compile` by directly accessing the `torch.Module` instances to create the `torch.Sequential`
- improved test library with more extensive testing of `RNNLayer` and `Network` functionalities

## 0.4

### 0.4.1

- added new pytests that test the functionalities of the `RNNLayer.record` and `RNNLayer.reset` methods
- added new pytests that test the initialization functions of `Network`
- improved integration of PyRates into RectiPy, by making sure that all PyRates caches are cleared, even if building the network functions fails due to erroneous user inputs

### 0.4.0

- removed all in-place operations for non-spiking networks
- changed pyrates interface such that vector-field updates are not performed in-place anymore
- only in-place operation left: Spike resetting
- added methods `Network.forward` and `Network.parameters` that allow the class `Network` to be embedded in larger network structures.
- added method `RNNLayer.reset` as a method that can be used to reset the state vector of the RNN
- added new tests for the rnn layer
- debugged `detach` method in rnn layer
- debugged issues with in-place operations and autograd
- added a new example for parameter fitting within the RNN layer

## 0.3

### 0.3.1

- improved documentation
- added pytests for the initialization functions of the rnn layer
- debugged index-finding functions for trainable parameters in the rnn layer
- improved integration of pyrates functions into rnn layer

### 0.3.0

- added utility function `readout` that allows to train a readout classifier on collected network states and targets
- added new gradient descent optimizer options
- added possibility of making an optimizer step only every `x` training steps (gradients will accumulate over these steps)

## 0.2

### 0.2.0

- renamed the model template package to avoid interference with the pyrates-intrinsic model template package
- added a utility function for the generation of input weight matrices
- added a utility function for winner-takes-all score calculation
- added getitem methods to the `Network` (integer-based indexing, returns layers) and `Observer` (string-based indexing, returns recordings) classes
- added the possibility to the `Network.train` method to train in epochs
- made the `device` argument of `Network.compile` optional
- ensured that the activation functions of the `OutputLayer` are always applied to the first dimension of the outputs

## 0.1

### 0.1.5

- ensured that state variable indices in RNN layer use correct data type (`torch.int64`)

### 0.1.4

- added pytests for the output layer
- added checks on the correctness of the input arguments for the output layer
- added keyword arguments to the `OutputLayer.__init__()` that are passed on to `torch.nn.Linear` if `trainable=True`

### 0.1.3

- added pytests for the input layer
- added a CircleCI config
- added automated execution of all tests via CircleCI upon pushing to github
- added `pytest` to the requirements

### 0.1.2

- added docstrings to the Network class for all non-private methods
- added docstrings to the Obsever class for all non-private methods
- made `Network.compile` a public method and reduced the number of automatized calls to it by `Network` (`Network.train`, `Network.test` and `Network.run` only call `Network.compile` themselves if it hasn't been done before)
- added a public property `Network.model` that provides read access to the pytorch model of the network

### 0.1.1

- added automated pypi releases
- added github workflow for pypi releases
- updated readme

### 0.1.0

- code structure: 
  - network class as main user interface
  - input, output, and rnn layers as network components
  - observer as class for results storage
- model templates package for yaml definition files
- installation instructions