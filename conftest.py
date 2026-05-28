from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parent
sys.path[:] = [
    path for path in sys.path if Path(path or ".").resolve() != _REPO_ROOT
]
sys.modules.pop("pyrxmesh", None)
importlib.import_module("pyrxmesh")

if torch.version.cuda is None:
    raise pytest.UsageError("PyRXMesh tests require CUDA-enabled PyTorch.")

if not torch.cuda.is_available():
    raise pytest.UsageError(
        "PyRXMesh tests require torch.cuda.is_available()."
    )
