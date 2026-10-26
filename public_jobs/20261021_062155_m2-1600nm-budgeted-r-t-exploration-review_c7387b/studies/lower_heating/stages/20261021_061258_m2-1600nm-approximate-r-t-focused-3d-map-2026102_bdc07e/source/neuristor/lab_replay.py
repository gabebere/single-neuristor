"""Compatibility import for neuristor.analysis.lab_replay.

New implementation belongs in analysis/lab_replay.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.analysis.lab_replay")
sys.modules[__name__] = _implementation
