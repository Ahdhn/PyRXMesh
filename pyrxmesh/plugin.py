"""Scaffold PyRXMesh custom CUDA plugin packages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _pyproject(module: str) -> str:
    dist_name = module.replace("_", "-")
    return f"""[build-system]
requires = [
    "scikit-build-core>=0.12",
    "pybind11>=2.13",
]
build-backend = "scikit_build_core.build"

[project]
name = "{dist_name}"
version = "0.1.0"
description = "Custom PyRXMesh CUDA kernels"
requires-python = ">=3.10"
dependencies = [
    "PyRXMesh",
]

[tool.scikit-build]
minimum-version = "0.12"
cmake.version = ">=3.24,<4"
cmake.build-type = "Release"
build-dir = "build/{{wheel_tag}}"
wheel.packages = ["src/{module}"]
build.targets = ["_{module}"]
install.components = ["python"]
"""


def _cmake(module: str) -> str:
    return f"""cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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


def _source(module: str) -> str:
    return f"""#include <pybind11/pybind11.h>

#include "pyrxmesh/plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;

void compute_edge_lengths(py::object mesh_obj,
                          py::object coords_obj,
                          py::object out_obj)
{{
    auto coords = pyrxmesh::vertex_attribute<float>(coords_obj);
    auto out    = pyrxmesh::edge_attribute<float>(out_obj);

    if (coords.get_num_attributes() < 3) {{
        throw std::runtime_error("coords must have at least 3 values per vertex.");
    }}
    if (out.get_num_attributes() != 1) {{
        throw std::runtime_error("out must have exactly 1 value per edge.");
    }}

    pyrxmesh::for_each<Op::EV, 256>(
        mesh_obj,
        [coords, out] __device__(const EdgeHandle& eh,
                                 const VertexIterator& iter) mutable {{
            const Eigen::Vector3f a = coords.to_eigen<3>(iter[0]);
            const Eigen::Vector3f b = coords.to_eigen<3>(iter[1]);
            out(eh) = (a - b).norm();
        }});
}}

PYBIND11_MODULE(_{module}, m)
{{
    pyrxmesh::require_compatible_runtime(m);

    m.def("compute_edge_lengths",
          &compute_edge_lengths,
          py::arg("mesh"),
          py::arg("coords"),
          py::arg("out"));
}}
"""


def _init(module: str) -> str:
    return f"""from __future__ import annotations

from ._{module} import compute_edge_lengths

__all__ = ["compute_edge_lengths"]
"""


def _readme(module: str) -> str:
    return f"""# {module}

Custom CUDA kernels for PyRXMesh.

## Layout

```text
{module}/
  pyproject.toml
  CMakeLists.txt
  src/
    {module}.cu             # CUDA source for the plugin
    {module}/__init__.py    # Python package source
  my_script.py              # your own scripts can live here
```

Python sources live under `src/`, so the plugin root never shadows the
installed package. Run scripts can sit at the plugin root and `import {module}`
directly with no `sys.path` setup.

## Build

Build inside the same conda environment where `pyrxmesh` is installed:

```bash
python -m pip install -v --no-build-isolation .
```

After installation, `{module}` is available from any directory in the env,
just like `pyrxmesh`.

## Use from Python

```python
import pyrxmesh as rx
import {module}

mesh = rx.RXMeshStatic("mesh.obj")
coords = mesh.input_vertex_coordinates()
edge_lengths = mesh.add_edge_attribute("edge_lengths", dtype="float32", dim=1)

{module}.compute_edge_lengths(mesh, coords, edge_lengths)
print(edge_lengths.to_numpy())
```
"""


def init_plugin(module: str, output_dir: Path, force: bool = False) -> Path:
    if not _MODULE_RE.match(module):
        raise ValueError(
            "Plugin module name must be a valid Python identifier, for example "
            "'my_kernels'."
        )

    root = output_dir / module
    root.mkdir(parents=True, exist_ok=True)

    _write(root / "pyproject.toml", _pyproject(module), force)
    _write(root / "CMakeLists.txt", _cmake(module), force)
    _write(root / "README.md", _readme(module), force)
    _write(root / "src" / f"{module}.cu", _source(module), force)
    _write(root / "src" / module / "__init__.py", _init(module), force)

    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a custom kernel plugin")
    init.add_argument("module", help="Python module name, for example my_kernels")
    init.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the plugin package directory will be created",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scaffold files",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        root = init_plugin(args.module, args.output_dir, args.force)
        print(root)


if __name__ == "__main__":
    main()
