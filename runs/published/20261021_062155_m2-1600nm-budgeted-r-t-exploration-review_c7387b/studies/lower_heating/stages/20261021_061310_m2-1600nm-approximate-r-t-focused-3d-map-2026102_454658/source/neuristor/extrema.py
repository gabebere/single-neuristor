"""Compatibility import for neuristor.analysis.extrema.

New implementation belongs in analysis/extrema.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.analysis.extrema")
sys.modules[__name__] = _implementation
