#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <cuda_runtime_api.h>

#include <pybind11/pybind11.h>

#include "bindings/common.h"
#include "bindings/dlpack_minimal.h"

#include "rxmesh/util/macros.h"

namespace py = pybind11;

namespace pyrxmesh_py::dlpack_util {

// ----------------------------------------------------------------------------
// dtype mapping
// ----------------------------------------------------------------------------

template <typename T>
inline dlpack::DLDataType dtype_for()
{
    if constexpr (std::is_same_v<T, float>) {
        return {2, 32, 1};
    } else if constexpr (std::is_same_v<T, double>) {
        return {2, 64, 1};
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return {0, 32, 1};
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return {0, 8, 1};
    } else {
        static_assert(always_false<T>::value,
                      "Unsupported DLPack dtype for PyRXMesh.");
    }
}

inline dlpack::DLDataType dtype_for(DType dtype)
{
    switch (dtype) {
        case DType::Float32:
            return {2, 32, 1};
        case DType::Float64:
            return {2, 64, 1};
        case DType::Int32:
            return {0, 32, 1};
        case DType::Int8:
            return {0, 8, 1};
        default:
            throw std::invalid_argument(
                "DLPack export encountered an unsupported dtype.");
    }
}

inline std::string dtype_to_name(dlpack::DLDataType dtype)
{
    if (dtype.lanes != 1) {
        throw std::invalid_argument(
            "PyRXMesh only supports DLPack tensors with lanes == 1.");
    }
    if (dtype.code == 2 && dtype.bits == 32) {
        return "float32";
    }
    if (dtype.code == 2 && dtype.bits == 64) {
        return "float64";
    }
    if (dtype.code == 0 && dtype.bits == 32) {
        return "int32";
    }
    if (dtype.code == 0 && dtype.bits == 8) {
        return "int8";
    }
    throw std::invalid_argument(
        "PyRXMesh DLPack import only supports float32, float64, int32, and "
        "int8 tensors.");
}

inline bool dtype_equal(dlpack::DLDataType a, dlpack::DLDataType b)
{
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}


// -----------------------------------------------------------------------------
// stream sentinels and synchronization
//
// The Array API DLPack contract uses sentinel stream values:
//   None / not provided -> producer chooses; we default to legacy default
//   stream 1                   -> per-thread default stream / "default" CUDA
//   stream -1                  -> caller asserts no synchronization needed
// Reference:
// https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.array.__dlpack__.html
// -----------------------------------------------------------------------------

inline py::object default_cuda_stream_arg()
{
    return py::int_(1);
}

inline bool is_cuda_no_sync_stream(py::object stream)
{
    return !stream.is_none() && cuda_stream_arg_value(std::move(stream)) == -1;
}

inline bool source_is_cuda(py::object source)
{
    if (!py::hasattr(source, "__dlpack_device__")) {
        return false;
    }
    py::tuple device = source.attr("__dlpack_device__")().cast<py::tuple>();
    if (py::len(device) < 1) {
        return false;
    }
    return device[0].cast<int>() == static_cast<int>(dlpack::kDLCUDA);
}

inline void synchronize_export_stream(py::object stream)
{
    using namespace rxmesh;
    if (is_cuda_no_sync_stream(stream)) {
        return;
    }

    cudaStream_t consumer_stream = parse_cuda_stream_arg(std::move(stream));
    if (consumer_stream == nullptr) {
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        return;
    }

    cudaEvent_t event = nullptr;
    CUDA_ERROR(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_ERROR(cudaEventRecord(event, nullptr));
    CUDA_ERROR(cudaStreamWaitEvent(consumer_stream, event, 0));
    CUDA_ERROR(cudaEventDestroy(event));
}


// -----------------------------------------------------------------------------
// Capsule lifecycle
// -----------------------------------------------------------------------------

/**
 * Owner-side capsule destructor. If the capsule is still in the original
 * "dltensor" state (i.e. the consumer never claimed it), invoke the manager's
 * deleter. If the consumer renamed it to "used_dltensor", the consumer now
 * owns the lifetime and we do nothing.
 */
inline void capsule_destructor(PyObject* capsule)
{
    if (PyCapsule_IsValid(capsule, "used_dltensor")) {
        return;
    }
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
        return;
    }
    auto* managed = static_cast<dlpack::DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed && managed->deleter) {
        managed->deleter(managed);
    }
}

/**
 * Generic deleter for `DLManagedTensor` whose `manager_ctx` points to a
 * heap-allocated context object of type `Context`. The caller is responsible
 * for setting `managed->manager_ctx` and ensuring `Context` was allocated
 * with `new`.
 */
template <typename Context>
inline void managed_tensor_deleter(dlpack::DLManagedTensor* self)
{
    if (!self) {
        return;
    }
    delete static_cast<Context*>(self->manager_ctx);
    delete self;
}


/**
 * Acquire a DLPack capsule from a Python object. Accepts either a raw
 * `dltensor` capsule or any object implementing `__dlpack__()`. For CUDA
 * sources, threads `stream` into the producer call (defaulting to the
 * Array-API CUDA default stream sentinel `1` if `stream` is None). For host
 * sources, no stream is sent.
 *
 * `api_name` is used only in error messages (e.g.
 * "DenseMatrix.from_dlpack_copy()").
 */
inline py::object acquire_capsule(py::object  source,
                                  py::object  stream,
                                  const char* api_name)
{
    if (PyCapsule_IsValid(source.ptr(), "dltensor")) {
        return source;
    }
    if (!py::hasattr(source, "__dlpack__")) {
        throw std::invalid_argument(
            std::string(api_name) +
            " expects a DLPack capsule or an object with __dlpack__().");
    }
    if (source_is_cuda(source)) {
        py::dict kwargs;
        kwargs["stream"] =
            stream.is_none() ? default_cuda_stream_arg() : stream;
        return source.attr("__dlpack__")(**kwargs);
    }
    return source.attr("__dlpack__")();
}

/**
 * Convenience overload that lets callers omit the stream and always passes
 * the default Array-API CUDA stream sentinel. Equivalent to the existing
 * dense-matrix `acquire_dlpack_capsule(source)` helper.
 */
inline py::object acquire_capsule(py::object source, const char* api_name)
{
    return acquire_capsule(std::move(source), py::none(), api_name);
}

/**
 * Mark a producer-supplied capsule as consumed and trigger its deleter.
 * Renames the capsule from "dltensor" to "used_dltensor" so the producer's
 * own destructor becomes a no-op, then invokes `managed->deleter` to release
 * any provider-side state. Must be called once the consumer has finished
 * copying out of the tensor.
 */
inline void mark_consumed(py::object capsule, dlpack::DLManagedTensor* managed)
{
    auto* deleter = managed ? managed->deleter : nullptr;
    PyCapsule_SetName(capsule.ptr(), "used_dltensor");
    PyCapsule_SetDestructor(capsule.ptr(), nullptr);
    if (deleter) {
        deleter(managed);
    }
}

/**
 * Read the underlying `DLManagedTensor*` from a `dltensor` capsule without
 * marking it consumed. Returns nullptr if the capsule is not a valid
 * `dltensor`.
 */
inline dlpack::DLManagedTensor* extract_managed(py::object capsule)
{
    if (!PyCapsule_IsValid(capsule.ptr(), "dltensor")) {
        return nullptr;
    }
    return static_cast<dlpack::DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
}


// -----------------------------------------------------------------------------
// `__dlpack_device__` helpers
// -----------------------------------------------------------------------------

/**
 * Build the `__dlpack_device__` tuple for an object whose primary data lives
 * on either DEVICE or HOST. Prefers DEVICE if both allocations exist (most
 * RXMesh objects own both, and DLPack export will copy/sync from DEVICE).
 */
inline py::tuple device_tuple(bool        is_device_allocated,
                              bool        is_host_allocated,
                              const char* api_name)
{
    using namespace rxmesh;
    if (is_device_allocated) {
        int device_id = 0;
        CUDA_ERROR(cudaGetDevice(&device_id));
        return py::make_tuple(static_cast<int>(dlpack::kDLCUDA), device_id);
    }
    if (is_host_allocated) {
        return py::make_tuple(static_cast<int>(dlpack::kDLCPU), 0);
    }
    throw std::runtime_error(std::string(api_name) +
                             " has no allocated memory.");
}

}  // namespace pyrxmesh_py::dlpack_util
