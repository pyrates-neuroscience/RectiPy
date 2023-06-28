Changelog
=========

0.10
----

0.10.3
------

- updated documentation of user interfaces
- updated use examples for user interfaces
- removed a minor bug in `nodes.py` that led to issues with retrieving variable indices from single-neuron nodes
- removed a bug in `nodes.py` where, in some cases, the state vector `y` was not moved to the correct device after state resets

0.10.2
------

- debugged the `Network.reset()` method. Properly resets the network to the passed state dictionary now, if provided.
- improved the `RateNet.reset()` method. The method can now handle both numpy arrays as well as tensors as input.

0.10.1
------

- added a new method `Network.clear()` that removes all nodes and edges from a `rectipy.Network` instance
- added new keyword arguments to `Network.detach()` that allow to customize the behavior of the method with respect to which tensors are going to be detached and whether they should require a gradient after detachment or not
- added a new method `Network.set_var()` that allows to change the values of variables in the network instance
- added a new method `Node.set_param()` that allows to change the values of parameters of the node
- improved documentation of the methods of `rectipy.Network`

0.10.0
------

- Changed user interface: `rectipy.Network` is now initialized as an empty graph. Nodes and edges can be added to it afternwards.
- Added new modules: `rectipy.nodes` and `rectipy.edges` provide different classes for nodes and edges that can be added to `rectipy.Network` instances
- Altered training functionalities: `rectipy.Network` now features two methods for parameter optimization: `Network.fit_ridge` for Ridge regression-based training of a set of readout weights, and `Network.fit_bptt` for gradient descent based on backpropagation through time.
- Implemented truncated backpropagation through time
- Implemented gradient surrogates for spiking neural networks
- New functionalities of the `rectipy.Observer` module: All recorded outputs can now be returned either as a list of `torch.Tensor` objects, as numpy arrays, or as `pandas.DataFrame` objects.
- Added convenience functions on `rectipy.Network` for adding and getting nodes, edges and network variables.
- Improved integration of `rectipy.Network` with pytorch parameter optimization methods by adding high-level methods for (i) detaching all state-variables from the current graph for gradient computation, and (ii) resetting the state of the entire network.
- Updated unit tests and documentation to work with the above described changes
- Updated PyRates interface to work with recent changes to the `pyrates.CircuitTemplate.add_edges_from_matrix` method

0.9
---

0.9.3
~~~~~

- added a new IK neuron template with biexponential synaptic dynamics
- improved layout of the readthedocs documentation website
- changed the `readout` function to use the `SGDClassifier` instead of `Ridge` from sklearn

0.9.2
~~~~~

- cleaned some code after model deployment changes made in 0.9.1
- debugged the `from_template` initialization methods

0.9.1
~~~~~

- minor bug fix of faulty normalization of input weights in `utility.input_connections`
- resolved issues with the model deployment on a certain device. Instead of providing the device ("cpu" or "cuda") to the `Network.compile` method, it is to be provided during initialization now
- debugged network initialization method `Network.from_template`

0.9.0
~~~~~

- debugged global recovery variable definition of izhikevich model template
- debugged simulation test
- added a new `rectipy.Network` initialization method: `Network.from_template` that allows to initialize `Network`
  instances from `pyrates.CircuitTemplate` instances. This way, the user has full control over the construction of the
  network template.

0.8
---

0.8.0
~~~~~

- added a use example for rectipy-torch integration
- added a function for matrix normalization to utility
- added the izhikevich neuron model as a template
- added an izhikevich neuron with global recovery variable as a template

0.7
---

0.7.0
~~~~~

- added visualization method `rectipy.observer.Observer.matshow` that allows to create 2D color-coded plots of multi-dimensional RNN state variables
- simplified alteration of default parameter values during network initialization
- added use example for training and testing via the `Network.train` and `Network.test` methods
- added a global coupling constant `k` to the qif model template
- improved docstrings

0.6
---

0.6.0
~~~~~

- added use example for the LIF neuron model
- new variable views available on the `rectipy.Network` and `rectipy.rnn_layer.RNNLayer` classes
- :code:`Network.__getitem__()` and :code:`RNNLayer.__getitem__()` allow to directly access parameters and variables of the `RNNLayer` instance
- integrated the new variable views into the documentation and testing suite
- simplified code for model definitions based on the new variable views

0.5
---

0.5.2
~~~~~

- added use example for the QIF neuron models
- added use example for the leaky-integrator rate neuron model
- added use example gallery skeleton
- added use example for network initialization
- added use example for numerical simulations
- added use example for the observer
- removed bug from SRNNLayer that caused model initialization to fail when no `dtype` for variales was provided
- removed bug from the sigmoid operator that is part of the `leaky_integrator.yaml` model definition file
- added `.gitignore` file
- added model template for LIF neurons
- improved docstrings of the `Network` class

0.5.1
~~~~~

- added documentation source files for a readthedocs documentation website
- added yaml configuration and config files for readthedocs installation
- added a first use example
- added installation instructions
- added the changelog to the readthedocs website sources
- added a full API section
- renamed the `tests` module to `rectipy_tests` to avoid confusion with the `PyRates.tests` module

0.5.0
~~~~~

-  reduced overhead of ``InputLayer`` and ``OutputLayer`` by making them
   return instances of ``torch.nn.Linear`` or
   ``rectipy.input_layer.LinearStatic`` upon initialization
-  reduced overhead of ``Network.compile`` by directly accessing the
   ``torch.Module`` instances to create the ``torch.Sequential``
-  improved test library with more extensive testing of ``RNNLayer`` and
   ``Network`` functionalities

0.4
---

0.4.1
~~~~~

-  added new pytests that test the functionalities of the
   ``RNNLayer.record`` and ``RNNLayer.reset`` methods
-  added new pytests that test the initialization functions of
   ``Network``
-  improved integration of PyRates into RectiPy, by making sure that all
   PyRates caches are cleared, even if building the network functions
   fails due to erroneous user inputs

0.4.0
~~~~~

-  removed all in-place operations for non-spiking networks
-  changed pyrates interface such that vector-field updates are not
   performed in-place anymore
-  only in-place operation left: Spike resetting
-  added methods ``Network.forward`` and ``Network.parameters`` that
   allow the class ``Network`` to be embedded in larger network
   structures.
-  added method ``RNNLayer.reset`` as a method that can be used to reset
   the state vector of the RNN
-  added new tests for the rnn layer
-  debugged ``detach`` method in rnn layer
-  debugged issues with in-place operations and autograd
-  added a new example for parameter fitting within the RNN layer

0.3
---

0.3.1
~~~~~

-  improved documentation
-  added pytests for the initialization functions of the rnn layer
-  debugged index-finding functions for trainable parameters in the rnn
   layer
-  improved integration of pyrates functions into rnn layer

0.3.0
~~~~~

-  added utility function ``readout`` that allows to train a readout
   classifier on collected network states and targets
-  added new gradient descent optimizer options
-  added possibility of making an optimizer step only every ``x``
   training steps (gradients will accumulate over these steps)

0.2
---

0.2.0
~~~~~

-  renamed the model template package to avoid interference with the
   pyrates-intrinsic model template package
-  added a utility function for the generation of input weight matrices
-  added a utility function for winner-takes-all score calculation
-  added getitem methods to the ``Network`` (integer-based indexing,
   returns layers) and ``Observer`` (string-based indexing, returns
   recordings) classes
-  added the possibility to the ``Network.train`` method to train in
   epochs
-  made the ``device`` argument of ``Network.compile`` optional
-  ensured that the activation functions of the ``OutputLayer`` are
   always applied to the first dimension of the outputs

0.1
---

0.1.5
~~~~~

-  ensured that state variable indices in RNN layer use correct data
   type (``torch.int64``)

0.1.4
~~~~~

-  added pytests for the output layer
-  added checks on the correctness of the input arguments for the output
   layer
-  added keyword arguments to the ``OutputLayer.__init__()`` that are
   passed on to ``torch.nn.Linear`` if ``trainable=True``

0.1.3
~~~~~

-  added pytests for the input layer
-  added a CircleCI config
-  added automated execution of all tests via CircleCI upon pushing to
   github
-  added ``pytest`` to the requirements

0.1.2
~~~~~

-  added docstrings to the Network class for all non-private methods
-  added docstrings to the Obsever class for all non-private methods
-  made ``Network.compile`` a public method and reduced the number of
   automatized calls to it by ``Network`` (``Network.train``,
   ``Network.test`` and ``Network.run`` only call ``Network.compile``
   themselves if it hasn’t been done before)
-  added a public property ``Network.model`` that provides read access
   to the pytorch model of the network

0.1.1
~~~~~

-  added automated pypi releases
-  added github workflow for pypi releases
-  updated readme

0.1.0
~~~~~

-  code structure:

   -  network class as main user interface
   -  input, output, and rnn layers as network components
   -  observer as class for results storage

-  model templates package for yaml definition files
-  installation instructions
