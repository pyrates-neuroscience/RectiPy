
# -*- coding: utf-8 -*-
#
#
# RectiPy software framework for recurrent neural network training in
# Python. See also: https://github.com/pyrates-neuroscience/RectiPy
#
# Copyright (C) 2022 Richard Gast.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast et al. (in preparation)
"""User interface for the initialization, training, and testing of networks with 3 layers:
Input, recurrent network, and output layer.
"""

__author__ = "Richard Gast"
__status__ = "Development"
__version__ = "0.10.2"

from .network import Network, Observer
from .utility import random_connectivity, circular_connectivity, input_connections, normalize
from .utility import wta_score
