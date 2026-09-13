# Releasing PyRXMesh

This checklist is for maintainers. The release workflow is `.github/workflows/wheels.yml`

## 1. Review Version and Compatibility Inputs

- Set `[project].version` in `pyproject.toml` to the release version. `CMakeLists.txt` reads this value, so there is no second PyRXMesh package version to update.
- Review `PYRXMESH_RXMESH_GIT_REPO` and `PYRXMESH_RXMESH_GIT_TAG` in `pyproject.toml`. Prefer an immutable commit or release tag for a reproducible package.
- Review `PYRXMESH_PLUGIN_ABI_VERSION` in `CMakeLists.txt`, but change it only when the compiled plugin capsule/API is incompatible with existing plugins. A normal PyRXMesh release does not imply an ABI bump.
- Review other options that form the plugin build-configuration tag, i.e., PyRXMesh version, RXMesh repository/ref, double precision, cuDSS, SuiteSparse, and CUDA architectures.
- Confirm that optional Python features remain optional. In particular, Polyscope belongs to the `viz` extra and is not a required runtime dependency or an RXMesh build dependency.

Any change to the package version or build tag requires rebuilding the example plugins used for release verification.

## 2. Verify a Local Build

Use the development environment and build the release candidate from the configured RXMesh ref, not a local source override:

```bash
conda env create -f environment.yml
conda activate PyRXMesh
python -m pip install -v --no-build-isolation .
python -m pip install -v --no-build-isolation --force-reinstall --no-deps examples/diff_energy_plugin
python -m pytest
```

Then verify the ordinary plugin SDK and both example entry points:

```bash
python -m pip install -v --no-build-isolation --force-reinstall --no-deps examples/custom_kernel_plugin
python examples/custom_kernel_plugin/run_edge_lengths.py --input meshes/sphere3.obj
python examples/diff_energy_plugin/run_diff_energy.py --input meshes/sphere3.obj
```

The example plugins must import without a compatibility warning and must link against the SDK installed by the release candidate rather than a separate RXMesh build.

## 3. Refresh Generated Type Information

Generate the native stub from the installed release-candidate extension and check the committed result:

```bash
python -m pip install -r tools/requirements-stubs.txt
python tools/generate_stubs.py
python tools/generate_stubs.py --check
```

Review changes to `pyrxmesh/_rxmesh.pyi`. The generated file should change only when the installed public native surface or a declaration in `tools/stub_overrides.pyi` changed. Confirm that `pyrxmesh/py.typed` and all handwritten `.pyi` files remain in the package tree.

## 4. Build and Inspect the Source Distribution

Build the sdist through:

```bash
python -m pip install build
python -m build --sdist
```

Inspect the archive before publishing. It must contain the source needed for a PyRXMesh build, both plugin examples, public documentation, tests, benchmarks, the `cmake/` and `include/` trees, and the `pyrxmesh/` typing files. It must not contain local build directories, compiled extensions, caches, or logs. The include/exclude specifications are in `pyproject.toml`.

## 5. GitHub Build

`.github/workflows/wheels.yml` currently builds:

- CPython 3.10, 3.11, and 3.12;
- Linux x86-64 and Windows AMD64 wheels;
- CUDA 12.6 (the Windows installer requests 12.6.3);
- CUDA architectures `75;80;86;89;90`;
- one source distribution.

The workflow inspects Linux wheels with `auditwheel` and Windows wheels with `delvewheel`, then uploads the wheels and sdist as GitHub artifacts. It does not run the repository's GPU test suite, so the local verification above is a
release requirement.

Pull requests build artifacts without publishing. A manual workflow dispatch can publish to TestPyPI or PyPI. A tag matching `v*` publishes to PyPI through the workflow's trusted-publishing identity—make sure the tag version matches
`[project].version` before creating it.

## 6. Verify the Published Package

From a clean supported environment:

1. Install the published wheel and import `pyrxmesh` from outside this source checkout.
2. Print `pyrxmesh.abi_version()` and `pyrxmesh.build_config_tag()` and compare them with the intended release inputs.
3. Confirm editor/type-checker discovery of the bundled `py.typed` marker and stubs.
4. Build at least one example plugin against the installed wheel to verify the shipped headers, RXMesh library, and `pyrxmesh` CMake package.
5. Install `PyRXMesh[viz]` separately to verify the optional visualization dependency without making it mandatory for an ordinary install.
