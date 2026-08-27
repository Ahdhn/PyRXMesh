from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping
from pathlib import Path


MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def pyproject(module: str, description: str) -> str:
    return f"""[build-system]
requires = [
    "scikit-build-core>=0.12",
    "pybind11>=2.13",
]
build-backend = "scikit_build_core.build"

[project]
name = "{module.replace('_', '-')}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"
dependencies = ["PyRXMesh"]

[tool.scikit-build]
minimum-version = "0.12"
cmake.version = ">=3.25,<4"
cmake.build-type = "Release"
build-dir = "build/{{wheel_tag}}"
wheel.packages = ["src/{module}"]
build.targets = ["_{module}"]
install.components = ["python"]
"""


def cmake(module: str) -> str:
    return f"""cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project({module} LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${{Python3_EXECUTABLE}}" -m pyrxmesh.cmake_dir
    OUTPUT_VARIABLE pyrxmesh_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

find_package(pyrxmesh CONFIG REQUIRED PATHS "${{pyrxmesh_DIR}}" NO_DEFAULT_PATH)
pyrxmesh_add_plugin(_{module} src/{module}.cu)

install(TARGETS _{module}
    LIBRARY DESTINATION {module} COMPONENT python
    RUNTIME DESTINATION {module} COMPONENT python
)
"""


def create_plugin(
    module: str,
    output_dir: Path,
    files: Mapping[str, str],
    *,
    force: bool = False,
) -> Path:
    if not MODULE_RE.match(module):
        raise ValueError(
            "Plugin module name must be a valid Python identifier, for "
            "example 'my_plugin'."
        )

    root = output_dir / module
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = root / relative
        if path.exists() and not force:
            raise FileExistsError(
                f"{path} already exists; pass --force to overwrite it"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    return root


def scaffold_main(
    argv: list[str] | None,
    *,
    description: str,
    help_text: str,
    example: str,
    initializer: Callable[[str, Path, bool], Path],
) -> None:
    parser = argparse.ArgumentParser(description=description)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help=help_text)
    init.add_argument(
        "module",
        help=f"Python module name, for example {example}")
    init.add_argument("--output-dir", type=Path, default=Path.cwd())
    init.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(initializer(args.module, args.output_dir, args.force))
