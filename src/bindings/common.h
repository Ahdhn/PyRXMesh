#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime_api.h>

#include "pyrxmesh/plugin_api.h"
#include "rxmesh/handle.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/types.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"

#include "bindings/dispatch.h"

#if USE_POLYSCOPE
#include "polyscope/polyscope.h"
#endif

namespace py = pybind11;

namespace pyrxmesh_py {

inline constexpr uint32_t    plugin_abi_version = PYRXMESH_PLUGIN_ABI_VERSION;
inline constexpr const char* build_config_tag   = PYRXMESH_BUILD_CONFIG;

inline rxmesh::locationT parse_location(int location)
{
    return static_cast<rxmesh::locationT>(location);
}

inline int64_t cuda_stream_arg_value(py::object stream)
{
    if (stream.is_none()) {
        return 1;
    }
    if (!py::isinstance<py::int_>(stream)) {
        throw py::type_error("CUDA stream must be an integer or None.");
    }
    return stream.cast<int64_t>();
}

inline cudaStream_t parse_cuda_stream_arg(py::object stream)
{
    const int64_t value = cuda_stream_arg_value(std::move(stream));
    if (value == 0) {
        throw std::invalid_argument(
            "CUDA stream value 0 is ambiguous and not supported.");
    }
    if (value == 1) {
        return nullptr;
    }
    if (value == 2) {
        return cudaStreamPerThread;
    }
    if (value > 2) {
        return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(value));
    }
    throw std::invalid_argument(
        "CUDA stream value must be None, 1, 2, or a "
        "positive raw stream pointer.");
}

inline void cuda_stream_synchronize_arg(py::object stream)
{
    using namespace rxmesh;
    CUDA_ERROR(cudaStreamSynchronize(parse_cuda_stream_arg(std::move(stream))));
}

inline void synchronize_device_transfer(rxmesh::locationT source,
                                        rxmesh::locationT target)
{
    using namespace rxmesh;
    if (((source | target) & rxmesh::DEVICE) == rxmesh::DEVICE) {
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
    }
}

inline rxmesh::layoutT parse_layout(int layout)
{
    return static_cast<rxmesh::layoutT>(layout);
}

inline void ensure_polyscope_available()
{
#if !USE_POLYSCOPE
    throw std::runtime_error(
        "RXMesh was built with RX_USE_POLYSCOPE=OFF; visualization is not "
        "available.");
#endif
}

inline void show_polyscope()
{
    ensure_polyscope_available();
#if USE_POLYSCOPE
    polyscope::show();
#endif
}

inline glm::fvec3 sequence_to_fvec3(const py::sequence& values,
                                    const char*         name)
{
    if (py::len(values) != 3) {
        throw std::invalid_argument(std::string(name) +
                                    " must contain exactly 3 values.");
    }
    return glm::fvec3(values[0].cast<float>(),
                      values[1].cast<float>(),
                      values[2].cast<float>());
}

inline py::array_t<float> vec3_to_numpy(const glm::vec3& value)
{
    py::array_t<float> out({static_cast<py::ssize_t>(3)});
    auto               view = out.mutable_unchecked<1>();
    view(0)                 = value[0];
    view(1)                 = value[1];
    view(2)                 = value[2];
    return out;
}

void register_module_core(py::module_& m);
void register_handles(py::module_& m);
void register_attribute(py::module_& m);
void register_dense_matrix(py::module_& m);
void register_geometry(py::module_& m);
void register_sparse_matrix(py::module_& m);
void register_solvers(py::module_& m);
void register_mesh(py::module_& m);
void register_diff_energy(py::module_& m);


py::object make_sparse_matrix_from_mesh(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    rxmesh::Op                            op,
    std::string                           dtype);
}  // namespace pyrxmesh_py
