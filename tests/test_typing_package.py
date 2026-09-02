from __future__ import annotations

import ast
from pathlib import Path

import pyrxmesh as rx


_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "pyrxmesh"
_STUBS = (
    _PACKAGE / "_typing.pyi",
    _PACKAGE / "_rxmesh.pyi",
    _PACKAGE / "__init__.pyi",
    _PACKAGE / "diff.pyi",
)
_INLINE_TYPED = (
    _PACKAGE / "diff.py",
    _PACKAGE / "_numpy_interop.py",
    _PACKAGE / "_scipy_interop.py",
    _PACKAGE / "_torch_interop.py",
)
_OVERRIDES = _ROOT / "tools" / "stub_overrides.pyi"

_LITERALS = {
    "LocationName": {"none", "host", "device", "all"},
    "LayoutName": {"soa", "aos", "aosoa"},
    "OpName": {
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
    },
    "DTypeName": {
        "float32",
        "float",
        "float64",
        "double",
        "int32",
        "int",
        "int8",
    },
    "DenseOrder": {
        "col_major",
        "column_major",
        "F",
        "row_major",
        "row",
        "C",
    },
    "MeshOrder": {"linear", "global"},
    "CopyPolicy": {"auto", "never"},
    "ElementName": {"vertex", "edge", "face"},
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
        feature_version=(3, 10),
    )


def _alias_values(tree: ast.Module, name: str) -> set[str]:
    aliases: dict[str, ast.expr | None] = {}
    for statement in tree.body:
        if isinstance(statement, ast.AnnAssign):
            if isinstance(statement.target, ast.Name):
                aliases[statement.target.id] = statement.value
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = statement.value

    assert name in aliases and aliases[name] is not None, (
        f"missing type alias {name}"
    )

    def resolve(alias: str, seen: set[str]) -> tuple[set[str], bool]:
        assert alias not in seen, f"cyclic type alias involving {alias}"
        value = aliases[alias]
        assert value is not None
        values = {
            node.value
            for node in ast.walk(value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        has_literal = any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "Literal"
            for node in ast.walk(value)
        )
        references = {
            node.id
            for node in ast.walk(value)
            if isinstance(node, ast.Name) and node.id in aliases
        }
        for reference in references:
            nested_values, nested_literal = resolve(
                reference, seen | {alias}
            )
            values |= nested_values
            has_literal |= nested_literal
        return values, has_literal

    values, has_literal = resolve(name, set())
    assert has_literal, f"{name} must resolve to typing.Literal"
    return values


def _root_stub_exports(tree: ast.Module) -> set[str]:
    exports: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef)):
            if not statement.name.startswith("_"):
                exports.add(statement.name)
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if alias.asname == alias.name and not alias.name.startswith("_"):
                    exports.add(alias.name)
    return exports


def _annotation_names(tree: ast.Module) -> set[str]:
    annotations: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)

    return {
        node.id.lstrip("_")
        for annotation in annotations
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name)
    }


def test_pep561_files_exist() -> None:
    assert (_PACKAGE / "py.typed").is_file(), "pyrxmesh/py.typed is missing"
    for path in _STUBS + _INLINE_TYPED + (_OVERRIDES,):
        assert path.is_file(), f"required typing source is missing: {path.name}"


def test_typing_sources_use_python_310_compatible_syntax() -> None:
    for path in _STUBS + _INLINE_TYPED + (_OVERRIDES,):
        _parse(path)


def test_root_stub_exports_match_runtime() -> None:
    exports = _root_stub_exports(_parse(_PACKAGE / "__init__.pyi"))
    assert "diff" in exports
    assert exports == set(rx.__all__)


def test_public_string_literals_are_complete() -> None:
    tree = _parse(_PACKAGE / "_typing.pyi")
    for alias, expected in _LITERALS.items():
        assert _alias_values(tree, alias) == expected

    declared = {
        statement.target.id
        for statement in tree.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    }
    assert {"LocationArg", "LayoutArg", "OpArg"} <= declared


def test_public_apis_use_shared_literal_aliases() -> None:
    annotation_names = _annotation_names(_parse(_PACKAGE / "_rxmesh.pyi"))
    annotation_names |= _annotation_names(_parse(_PACKAGE / "diff.py"))

    assert {
        "LocationArg",
        "LayoutArg",
        "OpArg",
        "DTypeName",
        "DenseOrder",
        "MeshOrder",
        "CopyPolicy",
        "ElementName",
    } <= annotation_names


def test_key_high_level_api_declarations_exist() -> None:
    native = _parse(_PACKAGE / "_rxmesh.pyi")
    native_classes = {
        node.name for node in native.body if isinstance(node, ast.ClassDef)
    }
    assert {
        "Attribute",
        "DenseMatrix",
        "RXMeshStatic",
        "ScalarEnergy",
        "SparseMatrix",
    } <= native_classes

    diff_functions = {
        node.name
        for node in _parse(_PACKAGE / "diff.py").body
        if isinstance(node, ast.FunctionDef)
    }
    assert {"empty_soa", "to_global_order", "to_linear_soa"} <= diff_functions


def test_native_overrides_are_smaller_than_generated_stub() -> None:
    assert len(_OVERRIDES.read_text(encoding="utf-8").splitlines()) < len(
        (_PACKAGE / "_rxmesh.pyi")
        .read_text(encoding="utf-8")
        .splitlines()
    )
