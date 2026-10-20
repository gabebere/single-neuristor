"""Compatibility import for neuristor.interfaces.playground.

New implementation belongs in interfaces/playground.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.interfaces.playground")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
