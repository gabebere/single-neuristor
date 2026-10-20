"""Compatibility import for neuristor.fitting.joint_inference.

New implementation belongs in fitting/joint_inference.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.fitting.joint_inference")
sys.modules[__name__] = _implementation
