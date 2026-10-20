"""Compatibility import for neuristor.orchestration.map_workflows.

New implementation belongs in orchestration/map_workflows.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.orchestration.map_workflows")
sys.modules[__name__] = _implementation
