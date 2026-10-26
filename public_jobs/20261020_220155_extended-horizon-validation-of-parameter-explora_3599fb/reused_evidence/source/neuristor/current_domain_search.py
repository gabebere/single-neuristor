"""Compatibility import for neuristor.fitting.current_domain_search.

New implementation belongs in fitting/current_domain_search.py. This aliases the same
module object, including private names and monkeypatches.
"""

from importlib import import_module
import sys

_implementation = import_module("neuristor.fitting.current_domain_search")
sys.modules[__name__] = _implementation
