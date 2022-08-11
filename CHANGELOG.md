# Changelog

## 0.1

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