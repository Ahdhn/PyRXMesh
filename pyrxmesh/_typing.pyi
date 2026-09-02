from typing import (
    Literal,
    Protocol,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
)

LocationName: TypeAlias = Literal["none", "host", "device", "all"]
LocationArg: TypeAlias = LocationName | int
LocationLike: TypeAlias = LocationArg

LayoutName: TypeAlias = Literal["soa", "aos", "aosoa"]
LayoutArg: TypeAlias = LayoutName | int
LayoutLike: TypeAlias = LayoutArg

OpName: TypeAlias = Literal[
    "v",
    "e",
    "f",
    "vv",
    "ve",
    "vf",
    "fv",
    "fe",
    "ff",
    "ev",
    "ee",
    "ef",
    "ev_diamond",
]
OpResultName: TypeAlias = OpName | Literal["invalid"]


class _LegacyOp(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


OpArg: TypeAlias = OpName | _LegacyOp

ElementName: TypeAlias = Literal["vertex", "edge", "face"]
MeshOrder: TypeAlias = Literal["linear", "global"]
CopyPolicy: TypeAlias = Literal["auto", "never"]
EnergyCopyPolicy: TypeAlias = CopyPolicy

AttributeDType: TypeAlias = Literal[
    "float32",
    "float",
    "float64",
    "double",
    "int32",
    "int",
    "int8",
]
DTypeName: TypeAlias = AttributeDType
NumericDType: TypeAlias = Literal[
    "float32",
    "float",
    "float64",
    "double",
    "int32",
    "int",
]
FloatDType: TypeAlias = Literal["float32", "float", "float64", "double"]
CanonicalDType: TypeAlias = Literal["float32", "float64", "int32", "int8"]
CanonicalNumericDType: TypeAlias = Literal["float32", "float64", "int32"]
CanonicalFloatDType: TypeAlias = Literal["float32", "float64"]

DenseOrder: TypeAlias = Literal[
    "col_major",
    "column_major",
    "F",
    "row_major",
    "row",
    "C",
]
CanonicalDenseOrder: TypeAlias = Literal["col_major", "row_major"]
Permutation: TypeAlias = Literal["none", "symrcm", "symamd", "nstdis"]

Stream: TypeAlias = int | None
IntLike: TypeAlias = SupportsInt | SupportsIndex
FloatLike: TypeAlias = SupportsFloat | SupportsIndex
Scalar: TypeAlias = int | float
Shape2D: TypeAlias = tuple[int, int]
BlockShape: TypeAlias = tuple[int, int]
