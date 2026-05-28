from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent
sys.path[:] = [
    path for path in sys.path if Path(path or ".").resolve() != _REPO_ROOT
]
sys.modules.pop("pyrxmesh", None)
importlib.import_module("pyrxmesh")
