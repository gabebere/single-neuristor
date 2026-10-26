"""Compatibility import for neuristor.analysis.experimental_waveforms.

New implementation belongs in analysis/experimental_waveforms.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.analysis.experimental_waveforms")
sys.modules[__name__] = _implementation
