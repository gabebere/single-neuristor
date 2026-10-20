"""Compatibility import for neuristor.fitting.resistance_custom_analysis.

New implementation belongs in fitting/resistance_custom_analysis.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.fitting.resistance_custom_analysis")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
