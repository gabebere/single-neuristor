"""Compatibility import for neuristor.analysis.oscillation_audit.

New implementation belongs in analysis/oscillation_audit.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.analysis.oscillation_audit")
sys.modules[__name__] = _implementation
