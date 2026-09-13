# Developing PyRXMesh

This page for contributors who build PyRXMesh, change its bindings, or work against an RXMesh branch or local checkout. If you only want to use PyRXMesh, start with [README.md](README.md).

## Quick Start

Create the development environment and build the current checkout:

```bash
conda env create -f environment.yml
conda activate PyRXMesh
python -m pip install -v --no-build-isolation .
```

The test suite imports the separately compiled differentiable energy example. Install that plugin against the build you just installed, then run the tests:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps examples/diff_energy_plugin
python -m pytest
```

See [Building and Testing](docs/development/BUILDING_AND_TESTING.md) for platform requirements, CMake options, tests, and build troubleshooting.

## Work Against a Local RXMesh Checkout

Sometimes, you may want to work change RXMesh first before changing PyRXMesh. In this case, you would like to work with local RXMesh checked in on your machine. In this case, use `PYRXMESH_RXMESH_SOURCE_DIR` while changing PyRXMesh and RXMesh together and PyRXMesh will be built against your local RXMesh:

```bash
python -m pip install -v --no-build-isolation . -Ccmake.define.PYRXMESH_RXMESH_SOURCE_DIR=path/to/RXMesh
```

## Organization

- `pyrxmesh/`: Python package, interop helpers, plugin scaffolders, and type declarations
- `src/bindings/`: pybind11 and CUDA bindings
- `include/pyrxmesh/`: public headers installed for external plugins
- `cmake/`: installed CMake package and build metadata templates
- `examples/custom_kernel_plugin/`: ordinary CUDA plugin example
- `examples/diff_energy_plugin/`: differentiable-energy plugin used by the autograd tests
- `tests/`: Python runtime, interop, solver, typing, and autodiff tests
- `tools/`: maintainer utilities, including type-stub generation
- `.github/workflows/wheels.yml`: wheel, source-distribution, and publishing workflow

### Type Stubs

`pyrxmesh/_rxmesh.pyi` is generated from an installed extension and `tools/stub_overrides.pyi` contains a small set of richer declarations that cannot be inferred from pybind11 signatures. 


## Developer References

- [Building and Testing](docs/development/BUILDING_AND_TESTING.md)
- [Plugin Internals](docs/development/PLUGIN_SDK.md)
- [Autodiff Internals](docs/development/AUTODIFF_INTERNALS.md)
- [Releasing PyRXMesh](docs/development/RELEASING.md), including the separate package version and plugin ABI policies
