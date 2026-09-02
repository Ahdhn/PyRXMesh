"""Generate PyRXMesh's native stub from bindings plus small overrides."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "pyrxmesh"
COMMITTED_STUB = PACKAGE_DIR / "_rxmesh.pyi"
OVERRIDES = ROOT / "tools" / "stub_overrides.pyi"
STUBGEN_VERSION = "2.5.5"
MODULE = "pyrxmesh._rxmesh"

# These are the two known pybind11 docstring issues. The enum mapping repairs
# LogLevel defaults; only the exact raw RXMesh type is ignored.
INVALID_EXPRESSION = r"^rxmesh::RXMeshStatic$"
ENUM_LOCATION = r"^LogLevel$:pyrxmesh._rxmesh.LogLevel"


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _subprocess_environment() -> dict[str, str]:
    env = os.environ.copy()
    entries = env.get("PYTHONPATH", "").split(os.pathsep)
    entries = [
        entry for entry in entries if entry and not _inside(Path(entry), ROOT)
    ]
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    else:
        env.pop("PYTHONPATH", None)
    return env


def _require_stubgen() -> None:
    try:
        installed = version("pybind11-stubgen")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Install the stub tools with "
            "'python -m pip install -r tools/requirements-stubs.txt'."
        ) from exc
    if installed != STUBGEN_VERSION:
        raise RuntimeError(
            f"pybind11-stubgen {STUBGEN_VERSION} is required; found "
            f"{installed}."
        )


def _installed_module(cwd: Path, env: dict[str, str]) -> dict[str, object]:
    code = """
import json
import pyrxmesh
import pyrxmesh._rxmesh as module

public = sorted(name for name in dir(module) if not name.startswith("_"))
classes = {}
for name in public:
    value = getattr(module, name)
    if isinstance(value, type):
        classes[name] = sorted({
            member
            for base in value.__mro__
            if base is not object
            for member in vars(base)
            if not member.startswith("_")
        })

print(json.dumps({
    "package": pyrxmesh.__file__,
    "module": module.__file__,
    "public": public,
    "classes": classes,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "Could not import the installed pyrxmesh._rxmesh. Install "
            "PyRXMesh first.\n" + result.stderr.strip()
        )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    for key in ("package", "module"):
        if _inside(Path(str(info[key])), ROOT):
            raise RuntimeError(
                "Stub generation must use an installed PyRXMesh from outside "
                f"the source checkout; {key} resolved to {info[key]}."
            )
    return info


def _run_stubgen(cwd: Path, env: dict[str, str], output: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "pybind11_stubgen",
        "--exit-code",
        "--ignore-invalid-expressions",
        INVALID_EXPRESSION,
        "--enum-class-locations",
        ENUM_LOCATION,
        "--output-dir",
        str(output),
        MODULE,
    ]
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _generated_stub(output: Path) -> Path:
    path = output / "pyrxmesh" / "_rxmesh" / "__init__.pyi"
    if not path.is_file():
        raise RuntimeError(f"pybind11-stubgen did not produce {path}")
    return path


def _node_name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign):
        names = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if len(names) == 1:
            return names[0]
    return None


def _is_import(node: ast.stmt) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def _merge_class(generated: ast.ClassDef, override: ast.ClassDef) -> None:
    """Replace only members named by a partial override class."""

    replacements: dict[str, list[ast.stmt]] = {}
    for node in override.body:
        if (name := _node_name(node)) is not None:
            replacements.setdefault(name, []).append(node)

    emitted: set[str] = set()
    body: list[ast.stmt] = []
    for node in generated.body:
        name = _node_name(node)
        if name not in replacements:
            body.append(node)
        elif name not in emitted:
            body.extend(replacements[name])
            emitted.add(name)
    for name, nodes in replacements.items():
        if name not in emitted:
            body.extend(nodes)
    generated.body = body


def _without_runtime_namespace_imports(nodes: list[ast.stmt]) -> list[ast.stmt]:
    """Location/Layout are values, not importable Python submodules."""

    cleaned: list[ast.stmt] = []
    for node in nodes:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is None
        ):
            names = [
                item
                for item in node.names
                if item.name not in {"Layout", "Location"}
            ]
            if names:
                node.names = names
                cleaned.append(node)
        else:
            cleaned.append(node)
    return cleaned


def _merge_stub(generated_path: Path, override_path: Path) -> str:
    generated = ast.parse(
        generated_path.read_text(encoding="utf-8"),
        filename=str(generated_path),
    )
    override = ast.parse(
        override_path.read_text(encoding="utf-8"),
        filename=str(override_path),
    )
    generated.body = _without_runtime_namespace_imports(generated.body)

    existing_imports = {
        ast.dump(node, include_attributes=False)
        for node in generated.body
        if _is_import(node)
    }
    extra_imports: list[ast.stmt] = []
    for node in override.body:
        if not _is_import(node):
            continue
        key = ast.dump(node, include_attributes=False)
        if key not in existing_imports:
            extra_imports.append(node)
            existing_imports.add(key)

    generated_classes = {
        node.name: node
        for node in generated.body
        if isinstance(node, ast.ClassDef)
    }
    override_groups: dict[str, list[ast.stmt]] = {}
    new_definitions: list[ast.stmt] = []
    for node in override.body:
        if _is_import(node):
            continue
        if isinstance(node, ast.ClassDef) and node.name in generated_classes:
            _merge_class(generated_classes[node.name], node)
            continue
        if (name := _node_name(node)) is not None:
            override_groups.setdefault(name, []).append(node)

    generated_names = {
        name
        for node in generated.body
        if (name := _node_name(node)) is not None
    }
    for name, nodes in override_groups.items():
        if name not in generated_names:
            new_definitions.extend(nodes)

    replaced: set[str] = set()
    merged_body: list[ast.stmt] = []
    for node in generated.body:
        name = _node_name(node)
        if name in override_groups and name in generated_names:
            if name not in replaced:
                merged_body.extend(override_groups[name])
                replaced.add(name)
        else:
            merged_body.append(node)

    insert_at = 0
    if (
        merged_body
        and isinstance(merged_body[0], ast.Expr)
        and isinstance(merged_body[0].value, ast.Constant)
        and isinstance(merged_body[0].value.value, str)
    ):
        insert_at = 1
    while insert_at < len(merged_body) and _is_import(merged_body[insert_at]):
        insert_at += 1
    merged_body[insert_at:insert_at] = extra_imports + new_definitions
    generated.body = merged_body
    ast.fix_missing_locations(generated)
    return ast.unparse(generated).rstrip() + "\n"


def _declared_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )
    return names


def _declared_classes(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    direct: dict[str, set[str]] = {}
    bases: dict[str, list[str]] = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        members = {
            name
            for item in node.body
            if (name := _node_name(item)) is not None
        }
        direct[node.name] = members
        bases[node.name] = [
            base.id for base in node.bases if isinstance(base, ast.Name)
        ]

    resolved: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> set[str]:
        if name in resolved:
            return resolved[name]
        if name not in direct or name in visiting:
            return set()
        visiting.add(name)
        members = set(direct[name])
        for base in bases[name]:
            members.update(resolve(base))
        visiting.remove(name)
        resolved[name] = members
        return members

    for name in direct:
        resolve(name)
    return resolved


def _check_output(
    stub: Path, public: set[str], classes: dict[str, set[str]]
) -> None:
    required = [stub, PACKAGE_DIR / "_typing.pyi", PACKAGE_DIR / "py.typed"]
    missing_files = [
        str(path.relative_to(ROOT)) for path in required if not path.is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "Missing committed typing files: " + ", ".join(missing_files)
        )

    for package_stub in sorted(PACKAGE_DIR.glob("*.pyi")):
        ast.parse(
            package_stub.read_text(encoding="utf-8"),
            filename=str(package_stub),
        )

    missing_names = sorted(public - _declared_names(stub))
    if missing_names:
        raise RuntimeError(
            f"{stub} is missing native public names: " + ", ".join(missing_names)
        )

    declared_classes = _declared_classes(stub)
    missing_members = {
        class_name: sorted(members - declared_classes.get(class_name, set()))
        for class_name, members in classes.items()
        if members - declared_classes.get(class_name, set())
    }
    if missing_members:
        groups = [
            f"{class_name}: {', '.join(members)}"
            for class_name, members in sorted(missing_members.items())
        ]
        raise RuntimeError(
            f"{stub} is missing native public class members:\n"
            + "\n".join(groups)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or validate the committed PyRXMesh native stub."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=COMMITTED_STUB,
        help="merged stub path (default: pyrxmesh/_rxmesh.pyi)",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=None,
        help="optionally retain pybind11-stubgen's unmodified output here",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and fail if the committed stub is stale",
    )
    args = parser.parse_args()

    try:
        _require_stubgen()
        env = _subprocess_environment()
        with tempfile.TemporaryDirectory(prefix="pyrxmesh-stubs-") as temp:
            outside = Path(temp)
            if _inside(outside, ROOT):
                raise RuntimeError(
                    "The temporary working directory is inside the checkout."
                )
            info = _installed_module(outside, env)
            raw_output = (
                args.raw_output_dir.resolve()
                if args.raw_output_dir is not None
                else outside / "raw"
            )
            _run_stubgen(outside, env, raw_output)
            merged = _merge_stub(_generated_stub(raw_output), OVERRIDES)
            output = args.output.resolve()
            classes = {
                str(name): set(members)
                for name, members in dict(info["classes"]).items()
            }
            if args.check:
                if not output.is_file() or output.read_text(encoding="utf-8") != merged:
                    raise RuntimeError(
                        f"{output} is stale; run 'python tools/generate_stubs.py'."
                    )
                _check_output(output, set(info["public"]), classes)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(merged, encoding="utf-8")
                _check_output(output, set(info["public"]), classes)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print("Committed typing files are current and cover the native surface.")
    else:
        print(f"Generated {args.output.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
