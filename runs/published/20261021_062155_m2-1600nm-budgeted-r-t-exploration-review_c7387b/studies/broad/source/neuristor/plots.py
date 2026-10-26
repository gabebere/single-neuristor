"""Compatibility import for neuristor.legacy.plots.

New implementation belongs in legacy/plots.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.legacy.plots")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
