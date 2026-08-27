"""Create an external PyRXMesh CUDA-kernel plugin."""

from __future__ import annotations

from pathlib import Path

from ._plugin_scaffold import cmake, create_plugin, pyproject, scaffold_main


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
    return f"""from ._{module} import compute_edge_lengths

__all__ = ["compute_edge_lengths"]
"""


def _readme(module: str) -> str:
    return f"""# {module}

Custom CUDA kernels for PyRXMesh.

Edit `src/{module}.cu`, then build in the environment containing PyRXMesh:

```bash
python -m pip install -v --no-build-isolation .
```

```python
import pyrxmesh as rx
import {module}

mesh = rx.RXMeshStatic("mesh.obj")
coords = mesh.input_vertex_coordinates()
lengths = mesh.add_edge_attribute("edge_lengths", dtype="float32", dim=1)
{module}.compute_edge_lengths(mesh, coords, lengths)
```

See PyRXMesh's `CUSTOM_CUDA_PLUGINS.md` for the complete authoring guide.
"""


def init_plugin(module: str, output_dir: Path, force: bool = False) -> Path:
    return create_plugin(
        module,
        output_dir,
        {
            "pyproject.toml": pyproject(module, "Custom PyRXMesh CUDA kernels"),
            "CMakeLists.txt": cmake(module),
            "README.md": _readme(module),
            f"src/{module}.cu": _source(module),
            f"src/{module}/__init__.py": _init(module),
        },
        force=force,
    )


def main(argv: list[str] | None = None) -> None:
    scaffold_main(
        argv,
        description=__doc__ or "",
        help_text="Create a custom kernel plugin",
        example="my_kernels",
        initializer=init_plugin,
    )


if __name__ == "__main__":
    main()
