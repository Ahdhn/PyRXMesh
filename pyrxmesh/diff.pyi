from collections.abc import Sequence

import torch

from ._rxmesh import (
    HessianSparseMatrix as HessianSparseMatrix,
    JacobianSparseMatrix as JacobianSparseMatrix,
    RXMeshStatic,
    ScalarEnergy as ScalarEnergy,
)
from ._typing import ElementName


def empty_soa(
    shape: Sequence[int],
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | int | None = ...,
    requires_grad: bool = ...,
) -> torch.Tensor: ...


def to_linear_soa(
    mesh: RXMeshStatic,
    values: torch.Tensor,
    *,
    element: ElementName = ...,
) -> torch.Tensor: ...


def to_global_order(
    mesh: RXMeshStatic,
    values: torch.Tensor,
    *,
    element: ElementName = ...,
) -> torch.Tensor: ...


__all__: list[str]
