"""Compatibility import for neuristor.analysis.metrics.

New implementation belongs in analysis/metrics.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.analysis.metrics")
sys.modules[__name__] = _implementation
