"""Compatibility import for neuristor.interfaces.dashboard.

New implementation belongs in interfaces/dashboard.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.interfaces.dashboard")
sys.modules[__name__] = _implementation
