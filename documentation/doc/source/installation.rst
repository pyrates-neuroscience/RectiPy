*****************************
Installation and Requirements
*****************************

Prerequisites
-------------

`RectiPy` has been build and tested for `Python >= 3.6`.
We recommend to use `Anaconda` to create a new python environment with `Python >= 3.6`.
After that, the installation instructions provided below should work independent of the operating system.

Dependencies
------------

`RectiPy` has the following hard dependencies:

- `torch`
- `pyrates`
- `numpy`
- `matplotlib`
- `scipy`

Following the installation instructions below, these packages will be installed automatically, if not already installed within the `Python` environment you are using.

Installation
------------

`RectiPy` can be installed via the `pip` command.  Simply run the following line from a terminal with the target Python
environment being activated:

.. code-block:: bash

   pip install rectipy


You can install optional (non-default) packages by specifying one or more options in brackets, e.g.:

.. code-block:: bash

   pip install pyrates[dev]


Currently, the only available option is `dev` (includes `pytest` and `bump2version`).

Alternatively, it is possible to clone this repository and run one of the following lines
from the directory in which the repository was cloned:

.. code-block:: bash

   python setup.py install

or

.. code-block:: bash

   pip install .[<options>]