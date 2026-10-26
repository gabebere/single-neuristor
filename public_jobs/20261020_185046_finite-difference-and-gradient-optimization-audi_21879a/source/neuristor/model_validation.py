"""Compatibility import for neuristor.analysis.model_validation.

New implementation belongs in analysis/model_validation.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.analysis.model_validation")
sys.modules[__name__] = _implementation
