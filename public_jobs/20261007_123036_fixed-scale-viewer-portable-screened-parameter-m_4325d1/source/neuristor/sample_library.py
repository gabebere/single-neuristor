"""Compatibility import for neuristor.legacy.sample_library.

New implementation belongs in legacy/sample_library.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.legacy.sample_library")
sys.modules[__name__] = _implementation
