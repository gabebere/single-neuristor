"""Compatibility import for neuristor.orchestration.artifact_catalog.

New implementation belongs in orchestration/artifact_catalog.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.orchestration.artifact_catalog")
sys.modules[__name__] = _implementation
