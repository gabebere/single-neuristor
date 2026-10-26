"""Compatibility import for neuristor.plotting.map_animations.

New implementation belongs in plotting/map_animations.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.plotting.map_animations")
sys.modules[__name__] = _implementation
