"""Compatibility import for neuristor.analysis.physical_maps.

New implementation belongs in analysis/physical_maps.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.analysis.physical_maps")
sys.modules[__name__] = _implementation
