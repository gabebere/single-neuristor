"""Compatibility import for neuristor.plotting.replay_plots.

New implementation belongs in plotting/replay_plots.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.plotting.replay_plots")
sys.modules[__name__] = _implementation
