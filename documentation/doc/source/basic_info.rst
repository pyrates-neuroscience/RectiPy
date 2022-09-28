*******************
General Information
*******************

Basic features of RectiPy
-------------------------

- Frontend:
   - implement models via the `PyRates <https://github.com/pyrates-neuroscience/PyRates>`_ frontend
   - choose between `YAML` templates or `Python` classes to define your RNN
   - choose between various pre-implemented neuron models or implement your custom neuron model
   - add synaptic dynamics and/or delayed coupling
   - implement spiking or rate-based neuron models
   - full control over the RNN parameters that can be trained: Choose between synaptic weights, membrane time constants of neurons, ...
   - run pre-implemented training and testing workflows via a single function call, OR use any :code:`rectipy.Network` as a single unit/layer within your own, custom `torch` code
   - track any state variable of your model with any temporal resolution during training/testing/simulation procedures

- Backend:
   - make full use of the `PyTorch <https://pytorch.org/>`_ backend
   - full support of `autograd` for your parameter optimization
   - use any loss function and optimization algorithm available in `torch`
   - deploy your model on different hardware

Reference
---------

If you use PyRates, please cite:

`Gast, R., Rose, D., Salomon, C., Möller, H. E., Weiskopf, N., & Knösche, T. R. (2019). PyRates-A Python framework for rate-based neural simulations. PloS one, 14(12), e0225900. <https://doi.org/10.1371/journal.pone.0225900>`_

Contact
-------

If you have questions, problems or suggestions regarding RectiPy, please contact `Richard Gast <https://www.richardgast.me>`_.

Contribute
----------

RectiPy is an open-source project that everyone is welcome to contribute to. Check out our `GitHub repository <https://github.com/pyrates-neuroscience/RectiPy>`_
for all the source code, open issues etc. and send us a pull request, if you would like to contribute something to our software.

Useful links
------------

`RectiPy` makes use of two essential Python tools:

- Frontend: `PyRates <https://github.com/pyrates-neuroscience/PyRates>`_
- Backend: `PyTorch <https://pytorch.org/>`_

Each of these two Python tools comes with an extensive documentation that is complementary to the content covered on this documentation website.
