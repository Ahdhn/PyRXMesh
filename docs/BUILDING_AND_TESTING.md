# Building and Testing PyRXMesh

## Requirements

PyRXMesh requires Python 3.10 or newer, CMake 3.25 through 3.x, an NVIDIA GPU, and a CUDA toolkit.

- Windows builds require the Visual Studio C++ toolchain supported by the installed CUDA toolkit.
- Linux builds require a GCC or Clang version supported by the installed CUDA toolkit.

The provided Conda environment has Python 3.12, CMake, Ninja, NumPy, PyTorch with CUDA 12.6 packages, pytest, SciPy, Polyscope, scikit-build-core, and pybind11:

```bash
conda env create -f environment.yml
conda activate PyRXMesh
```

## Build the Extension

An isolated build uses the RXMesh repository and ref configured in `pyproject.toml`:

```bash
python -m pip install -v .
```

Once the development dependencies are installed, reuse the active environment for a faster inner loop:

```bash
python -m pip install -v --no-build-isolation .
```

scikit-build-core reuses `build/{wheel_tag}`. Repeating a build with the same Python, platform, RXMesh source, and CMake settings is incremental. Remove only the matching build directory when switching native configuration or when a cached object no longer represents the selected RXMesh source.

## Select the RXMesh Source

The default repository and ref are source-controlled under `[tool.scikit-build.cmake.define]` in `pyproject.toml`. Override a tag, branch, or commit for a local build with:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

Override both fields to use a fork:

```bash
python -m pip install -v . -Ccmake.define.PYRXMESH_RXMESH_GIT_REPO=https://github.com/<user>/RXMesh.git -Ccmake.define.PYRXMESH_RXMESH_GIT_TAG=<tag-or-commit>
```

Use a local checkout directly while developing RXMesh and PyRXMesh together:

```bash
python -m pip install -v --no-build-isolation . -Ccmake.define.PYRXMESH_RXMESH_SOURCE_DIR=path/to/RXMesh
```

## CMake Configuration

The supported PyRXMesh cache variables are:

| Variable                      |                  Default | Purpose                                                 |
| ----------------------------- | -----------------------: | ------------------------------------------------------- |
| `PYRXMESH_RXMESH_GIT_REPO`    | RXMesh GitHub repository | RXMesh fetch source                                     |
| `PYRXMESH_RXMESH_GIT_TAG`     |                   `main` | RXMesh branch, tag, or commit                           |
| `PYRXMESH_RXMESH_SOURCE_DIR`  |                    empty | Local RXMesh checkout                                   |
| `PYRXMESH_BUILD_RXMESH_TESTS` |                    `OFF` | Configure upstream RXMesh test targets                  |
| `PYRXMESH_BUILD_RXMESH_APPS`  |                    `OFF` | Configure upstream RXMesh application targets           |
| `PYRXMESH_USE_DOUBLE`         |                    `OFF` | Use double-precision RXMesh coordinate input            |
| `PYRXMESH_USE_CUDSS`          |                    `OFF` | Enable RXMesh cuDSS support                             |
| `PYRXMESH_USE_SUITESPARSE`    |                    `OFF` | Enable RXMesh SuiteSparse support                       |
| `PYRXMESH_INSTALL_PLUGIN_SDK` |                     `ON` | Install headers, libraries, and CMake files for plugins |

Pass a variable with `-Ccmake.define.NAME=value`. The compatibility tag records only the configured repository/ref, precision, optional solvers, CUDA architectures, version, and ABI. 

If `CMAKE_CUDA_ARCHITECTURES` is not set, a local build uses `native`. The release workflow currently builds `75;80;86;89;90`:

```bash
python -m pip install -v . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```

## Run the Tests

The full test suite imports `rxmesh_diff_energy` during collection, so you need to install the separately built example plugin against that exact runtime:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps examples/diff_energy_plugin
python -m pytest
```

Or test specific parts:

```bash
python -m pytest -v tests/test_core_mesh.py
python -m pytest -v tests/test_attributes.py
python -m pytest -v tests/test_diff_autograd.py
python -m pytest -v tests/test_typing_package.py
```

Build and run the ordinary plugin example when changing, e.g., the installed plugin SDK:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps examples/custom_kernel_plugin
python examples/custom_kernel_plugin/run_edge_lengths.py --input meshes/sphere3.obj
```

For a differentiable plugin testing:

```bash
python examples/diff_energy_plugin/run_diff_energy.py --input meshes/sphere3.obj
```

## Refresh and Check Type Information

The native stub must be generated from the installed extension after a public binding change:

```bash
python -m pip install -r tools/requirements-stubs.txt
python tools/generate_stubs.py
python tools/generate_stubs.py --check
```

The generator writes `pyrxmesh/_rxmesh.pyi`. Maintain only declarations that need manual refinement in `tools/stub_overrides.pyi`; Python helpers keep their annotations in their source or handwritten `.pyi` file.

## Build Troubleshooting

### A plugin reports a build configuration mismatch

The plugin was compiled against a different installed PyRXMesh. Rebuild it after reinstalling PyRXMesh. Changes to RXMesh template headers, `include/pyrxmesh/plugin_api.h`, `include/pyrxmesh/diff_plugin_api.h`, CUDA architectures, precision, or optional native libraries can require this even when the numeric plugin ABI version is unchanged.

### A rebuild appears to use old RXMesh code

Confirm the requested repository, ref, or `PYRXMESH_RXMESH_SOURCE_DIR`, then remove only the active `build/{wheel_tag}` directory and rebuild. Plugin projects maintain their own build directories and may need a clean rebuild as well.

### Import uses an unexpected package

Print `pyrxmesh.__file__`, `pyrxmesh._rxmesh.__file__`, `pyrxmesh.abi_version()`, and `pyrxmesh.build_config_tag()`. Run import checks from outside a plugin source directory so an unbuilt package cannot shadow the installed extension.
