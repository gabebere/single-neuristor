"""Compatibility import for neuristor.plotting.visualization.

New implementation belongs in plotting/visualization.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.plotting.visualization")
sys.modules[__name__] = _implementation
