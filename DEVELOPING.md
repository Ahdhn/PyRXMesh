# Developing PyRXMesh

This guide is for building PyRXMesh from source, testing changes, and working against RXMesh branches, forks, or local checkouts. If you only want to use PyRXMesh from Python, start with [README.md](README.md).

## Prerequisites

You need the platform CUDA/C++ toolchain:

- Windows: Visual Studio C++ tools plus NVIDIA CUDA Toolkit with `nvcc`.
- Linux: GCC/Clang compatible with your CUDA Toolkit, NVIDIA CUDA Toolkit with
  `nvcc`, and the OpenGL/X11 development packages required by Polyscope.

## Conda Environment

```bash
conda env create -f environment.yml
conda activate PyRXMesh
```

Install build requirements into the active environment when you want faster developer rebuilds without build isolation:

```bash
python -m pip install scikit-build-core pybind11 numpy
```

## Build From Source

The normal developer build uses the local default CUDA architecture behavior and the RXMesh ref configured in `pyproject.toml`:

```bash
python -m pip install -v .
```

For a faster inner loop after installing build requirements:

```bash
python -m pip install -v --no-build-isolation .
```

## Select An RXMesh Source

For release packages, the RXMesh ref should be source-controlled in `pyproject.toml`:

```toml
[tool.scikit-build.cmake.define]
PYRXMESH_RXMESH_GIT_TAG = "main"
```

For local experiments, override the RXMesh tag or commit from the command line:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

Examples:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=main
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=v0.2.1
```

To build against an RXMesh fork:

```bash
python -m pip install -v . \
  -Ccmake.define.PYRXMESH_RXMESH_GIT_REPO=https://github.com/<user>/RXMesh.git \
  -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

For local RXMesh development without fetching from Git:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_SOURCE_DIR=C:/path/to/RXMesh
```

## Release-Like Local Build

Release wheels build with Polyscope enabled and a broad CUDA architecture set:

```bash
python -m pip install -v . \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -Ccmake.define.PYRXMESH_USE_POLYSCOPE=ON
```

## Incremental Builds

Builds reuse a persistent scikit-build-core build directory:

```text
build/{wheel_tag}
```

The first install for a Python/platform/RXMesh configuration is a full build. Subsequent installs with the same settings should be incremental.

To force a clean rebuild, delete the matching `build/<wheel_tag>` directory.

## Run Examples

Examples are not part of the installed package, so editing them does not require reinstalling/recompiling PyRXMesh:

```bash
python examples/load_and_show.py --input mesh.obj
```

## Custom CUDA Plugins

Most PyRXMesh workflows should stay in Python. For RXMesh operations that require CUDA device lambdas, PyRXMesh supports small compiled plugin packages that operate on PyRXMesh meshes and attributes without extra Python-side copies.

See [CUSTOM_CUDA_PLUGINS.md](CUSTOM_CUDA_PLUGINS.md) for the plugin workflow.