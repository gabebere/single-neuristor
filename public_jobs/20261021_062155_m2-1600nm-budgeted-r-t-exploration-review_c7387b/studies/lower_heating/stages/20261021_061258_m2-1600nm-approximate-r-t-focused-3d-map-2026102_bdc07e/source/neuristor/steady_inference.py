"""Compatibility import for neuristor.fitting.steady_inference.

New implementation belongs in fitting/steady_inference.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.fitting.steady_inference")
sys.modules[__name__] = _implementation
