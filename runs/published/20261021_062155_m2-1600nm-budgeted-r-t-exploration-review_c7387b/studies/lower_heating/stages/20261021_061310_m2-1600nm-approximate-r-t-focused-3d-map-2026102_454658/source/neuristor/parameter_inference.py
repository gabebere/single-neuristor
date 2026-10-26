"""Compatibility import for neuristor.fitting.parameter_inference.

New implementation belongs in fitting/parameter_inference.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.fitting.parameter_inference")
sys.modules[__name__] = _implementation
