# Changelog

## 0.3

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