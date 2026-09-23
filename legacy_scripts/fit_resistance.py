from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neuristor.resistance_custom_analysis import *  # noqa: F401,F403
from neuristor.resistance_custom_analysis import main


if __name__ == "__main__":
    main()
