"""Compatibility import for neuristor.orchestration.sample_study.

New implementation belongs in orchestration/sample_study.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.orchestration.sample_study")
sys.modules[__name__] = _implementation
