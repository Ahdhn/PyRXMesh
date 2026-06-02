#pragma once

#include "bindings/dlpack_minimal.h"
#include "bindings/py_dense_matrix.h"

#include <cuda_runtime_api.h>
#include <cmath>

#include "rxmesh/matrix/dense_matrix.h"

namespace pyrxmesh_py {

template <typename T>
inline dlpack::DLDataType dlpack_dtype()
{
    if constexpr (std::is_same_v<T, float>) {
        return {2, 32, 1};
    } else if constexpr (std::is_same_v<T, double>) {
        return {2, 64, 1};
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return {0, 32, 1};
    } else {
        static_assert(always_false<T>::value, "Unsupported DenseMatrix dtype");
    }
}

inline std::string dtype_name_from_dlpack(dlpack::DLDataType dtype)
{
    if (dtype.code == 2 && dtype.bits == 32 && dtype.lanes == 1) {
        return "float32";
    }
    if (dtype.code == 2 && dtype.bits == 64 && dtype.lanes == 1) {
        return "float64";
    }
    if (dtype.code == 0 && dtype.bits == 32 && dtype.lanes == 1) {
        return "int32";
    }
    throw std::invalid_argument(
        "DenseMatrix.from_dlpack_copy() supports float32, float64, and int32 "
        "tensors.");
}

struct DlpackContext
{
    std::shared_ptr<PyDenseMatrix> owner;
    int64_t                        shape[2];
    int64_t                        strides[2];
};

inline bool is_cuda_dlpack_no_sync_stream(py::object stream)
{
    //https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.array.__dlpack__.html
    return !stream.is_none() && cuda_stream_arg_value(std::move(stream)) == -1;
}

inline py::object default_cuda_dlpack_stream_arg()
{
    return py::int_(1);
}

inline bool source_dlpack_device_is_cuda(py::object source)
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

inline void synchronize_dense_dlpack_export_stream(py::object stream)
{
    using namespace rxmesh;
    if (is_cuda_dlpack_no_sync_stream(stream)) {
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

template <typename T>
__global__ void copy_dlpack_to_dense_col_major_kernel(T*       dst,
                                                      const T* src,
                                                      int64_t  rows,
                                                      int64_t  cols,
                                                      int64_t  stride0,
                                                      int64_t  stride1)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n   = rows * cols;
    if (idx >= n) {
        return;
    }

    const int64_t row = idx % rows;
    const int64_t col = idx / rows;
    dst[idx]          = src[row * stride0 + col * stride1];
}

inline void dense_dlpack_capsule_destructor(PyObject* capsule)
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

inline void dense_dlpack_managed_tensor_deleter(dlpack::DLManagedTensor* self)
{
    delete static_cast<DlpackContext*>(self->manager_ctx);
    delete self;
}

inline py::capsule dense_matrix_to_dlpack(std::shared_ptr<PyDenseMatrix> self,
                                          int        location,
                                          py::object stream)
{
    using namespace rxmesh;
    const auto loc = parse_location(location);
    if (loc != rxmesh::HOST && loc != rxmesh::DEVICE) {
        throw std::invalid_argument(
            "DenseMatrix.to_dlpack() location must be Location.HOST or "
            "Location.DEVICE.");
    }
    if (loc == rxmesh::HOST && !self->is_host_allocated()) {
        throw std::runtime_error(
            "DenseMatrix.to_dlpack() requires an existing HOST allocation.");
    }
    if (loc == rxmesh::DEVICE && !self->is_device_allocated()) {
        throw std::runtime_error(
            "DenseMatrix.to_dlpack() requires an existing DEVICE allocation.");
    }

    if (loc == rxmesh::HOST) {
        if (!stream.is_none()) {
            throw std::invalid_argument(
                "DenseMatrix.to_dlpack(Location.HOST) requires stream=None.");
        }
    } else {
        synchronize_dense_dlpack_export_stream(std::move(stream));
    }

    auto* managed  = new dlpack::DLManagedTensor();
    auto* context  = new DlpackContext();
    context->owner = std::move(self);

    std::visit(
        [&](const auto& mat) {
            using MatT          = std::decay_t<decltype(mat)>;
            using T             = typename MatT::element_type::Type;
            context->shape[0]   = mat->rows();
            context->shape[1]   = mat->cols();
            context->strides[0] = 1;
            context->strides[1] = mat->rows();

            int device_id = 0;
            if (loc == rxmesh::DEVICE) {
                CUDA_ERROR(cudaGetDevice(&device_id));
            }

            managed->dl_tensor.data   = mat->data(loc);
            managed->dl_tensor.device = {
                loc == rxmesh::DEVICE ? dlpack::kDLCUDA : dlpack::kDLCPU,
                device_id};
            managed->dl_tensor.ndim        = 2;
            managed->dl_tensor.dtype       = dlpack_dtype<T>();
            managed->dl_tensor.shape       = context->shape;
            managed->dl_tensor.strides     = context->strides;
            managed->dl_tensor.byte_offset = 0;
        },
        context->owner->matrix);

    managed->manager_ctx = context;
    managed->deleter     = dense_dlpack_managed_tensor_deleter;

    return py::capsule(managed, "dltensor", dense_dlpack_capsule_destructor);
}

inline py::tuple dense_matrix_dlpack_device(const PyDenseMatrix& self)
{
    using namespace rxmesh;
    if (self.is_device_allocated()) {
        int device_id = 0;
        CUDA_ERROR(cudaGetDevice(&device_id));
        return py::make_tuple(static_cast<int>(dlpack::kDLCUDA), device_id);
    }
    if (self.is_host_allocated()) {
        return py::make_tuple(static_cast<int>(dlpack::kDLCPU), 0);
    }
    throw std::runtime_error("DenseMatrix has no allocated memory.");
}

inline py::capsule dense_matrix_dunder_dlpack(
    std::shared_ptr<PyDenseMatrix> self,
    py::object                     stream)
{
    const int loc = self->is_device_allocated() ?
                        static_cast<int>(rxmesh::DEVICE) :
                        static_cast<int>(rxmesh::HOST);
    return dense_matrix_to_dlpack(std::move(self), loc, std::move(stream));
}

inline py::object acquire_dlpack_capsule(py::object source)
{
    if (PyCapsule_IsValid(source.ptr(), "dltensor")) {
        return source;
    }
    if (!py::hasattr(source, "__dlpack__")) {
        throw std::invalid_argument(
            "DenseMatrix.from_dlpack_copy() expects a DLPack capsule or an "
            "object with __dlpack__().");
    }
    if (source_dlpack_device_is_cuda(source)) {
        py::dict kwargs;
        kwargs["stream"] = default_cuda_dlpack_stream_arg();
        return source.attr("__dlpack__")(**kwargs);
    }
    return source.attr("__dlpack__")();
}

inline void mark_dlpack_capsule_consumed(py::object               capsule,
                                         dlpack::DLManagedTensor* managed)
{
    auto* deleter = managed ? managed->deleter : nullptr;
    PyCapsule_SetName(capsule.ptr(), "used_dltensor");
    PyCapsule_SetDestructor(capsule.ptr(), nullptr);
    if (deleter) {
        deleter(managed);
    }
}

template <typename T>
inline void copy_dlpack_to_dense_matrix_typed(PyDenseMatrix&          output,
                                              const dlpack::DLTensor& tensor,
                                              rxmesh::locationT       location,
                                              int64_t                 stride0,
                                              int64_t                 stride1,
                                              cudaStream_t            copy_stream)
{
    using namespace rxmesh;
    using MatPtr = std::shared_ptr<rxmesh::DenseMatrix<T, Eigen::ColMajor>>;

    if (!std::holds_alternative<MatPtr>(output.matrix)) {
        throw std::invalid_argument(
            "DenseMatrix.from_dlpack_copy() internal dtype mismatch.");
    }

    auto& mat_ptr = std::get<MatPtr>(output.matrix);
    if (!mat_ptr) {
        throw std::runtime_error(
            "DenseMatrix.from_dlpack_copy() encountered an empty matrix.");
    }

    auto& mat = *mat_ptr;
    auto* src = reinterpret_cast<const T*>(
        static_cast<const char*>(tensor.data) + tensor.byte_offset);
    if (location == rxmesh::HOST) {
        for (int64_t j = 0; j < tensor.shape[1]; ++j) {
            for (int64_t i = 0; i < tensor.shape[0]; ++i) {
                mat(static_cast<int>(i), static_cast<int>(j)) =
                    src[i * stride0 + j * stride1];
            }
        }
    } else {
        constexpr int threads = 256;
        const int64_t n       = tensor.shape[0] * tensor.shape[1];
        copy_dlpack_to_dense_col_major_kernel<T>
            <<<static_cast<int>((n + threads - 1) / threads),
               threads,
               0,
               copy_stream>>>(
                mat.data(rxmesh::DEVICE),
                src,
                tensor.shape[0],
                tensor.shape[1],
                stride0,
                stride1);
        CUDA_ERROR(cudaGetLastError());
        CUDA_ERROR(cudaStreamSynchronize(copy_stream));
    }
}

inline std::shared_ptr<PyDenseMatrix> dense_matrix_from_dlpack_copy(
    py::object source)
{
    py::object capsule = acquire_dlpack_capsule(std::move(source));
    auto*      managed = static_cast<dlpack::DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    if (!managed) {
        PyErr_Clear();
        throw std::invalid_argument(
            "DenseMatrix.from_dlpack_copy() received an invalid DLPack "
            "capsule.");
    }

    try {
        const dlpack::DLTensor& tensor = managed->dl_tensor;
        if (tensor.ndim != 2) {
            throw std::invalid_argument(
                "DenseMatrix.from_dlpack_copy() expects a 2D tensor.");
        }
        if (tensor.shape[0] <= 0 || tensor.shape[1] <= 0) {
            throw std::invalid_argument(
                "DenseMatrix.from_dlpack_copy() expects positive tensor "
                "dimensions.");
        }

        rxmesh::locationT location;
        if (tensor.device.device_type == dlpack::kDLCPU) {
            location = rxmesh::HOST;
        } else if (tensor.device.device_type == dlpack::kDLCUDA) {
            location = rxmesh::DEVICE;
        } else {
            throw std::invalid_argument(
                "DenseMatrix.from_dlpack_copy() supports CPU and CUDA DLPack "
                "tensors.");
        }

        const std::string dtype = dtype_name_from_dlpack(tensor.dtype);
        const int64_t     stride0 =
            tensor.strides ? tensor.strides[0] : tensor.shape[1];
        const int64_t stride1 = tensor.strides ? tensor.strides[1] : 1;

        auto output =
            std::make_shared<PyDenseMatrix>(dtype,
                                            static_cast<int>(tensor.shape[0]),
                                            static_cast<int>(tensor.shape[1]),
                                            static_cast<int>(location),
                                            "col_major");
        const cudaStream_t copy_stream = nullptr;

        if (dtype == "float32") {
            copy_dlpack_to_dense_matrix_typed<float>(
                *output, tensor, location, stride0, stride1, copy_stream);
        } else if (dtype == "float64") {
            copy_dlpack_to_dense_matrix_typed<double>(
                *output, tensor, location, stride0, stride1, copy_stream);
        } else if (dtype == "int32") {
            copy_dlpack_to_dense_matrix_typed<int32_t>(
                *output, tensor, location, stride0, stride1, copy_stream);
        }
        mark_dlpack_capsule_consumed(capsule, managed);
        return output;
    } catch (...) {
        mark_dlpack_capsule_consumed(capsule, managed);
        throw;
    }
}

}  // namespace pyrxmesh_py
