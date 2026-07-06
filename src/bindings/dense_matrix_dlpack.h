#pragma once

#include "bindings/dispatch.h"
#include "bindings/dlpack_minimal.h"
#include "bindings/dlpack_utils.h"
#include "bindings/py_dense_matrix.h"

#include <cuda_runtime_api.h>
#include <cmath>

#include "rxmesh/matrix/dense_matrix.h"

namespace pyrxmesh_py {

struct DlpackContext
{
    std::shared_ptr<PyDenseMatrix> owner;
    int64_t                        shape[2];
    int64_t                        strides[2];
};

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
        dlpack_util::synchronize_export_stream(std::move(stream));
    }

    auto* managed  = new dlpack::DLManagedTensor();
    auto* context  = new DlpackContext();
    context->owner = std::move(self);

    with_typed_dense_matrix(*context->owner, [&](auto& typed) {
        using TypedT = std::decay_t<decltype(typed)>;
        using T      = typename TypedT::MatT::Type;
        auto& mat    = *typed.matrix;

        context->shape[0]   = mat.rows();
        context->shape[1]   = mat.cols();
        context->strides[0] = 1;
        context->strides[1] = mat.rows();

        int device_id = 0;
        if (loc == rxmesh::DEVICE) {
            CUDA_ERROR(cudaGetDevice(&device_id));
        }

        managed->dl_tensor.data   = mat.data(loc);
        managed->dl_tensor.device = {
            loc == rxmesh::DEVICE ? dlpack::kDLCUDA : dlpack::kDLCPU,
            device_id};
        managed->dl_tensor.ndim        = 2;
        managed->dl_tensor.dtype       = dlpack_util::dtype_for<T>();
        managed->dl_tensor.shape       = context->shape;
        managed->dl_tensor.strides     = context->strides;
        managed->dl_tensor.byte_offset = 0;
    });

    managed->manager_ctx = context;
    managed->deleter     = dlpack_util::managed_tensor_deleter<DlpackContext>;

    return py::capsule(managed, "dltensor", dlpack_util::capsule_destructor);
}

inline py::tuple dense_matrix_dlpack_device(const PyDenseMatrix& self)
{
    return dlpack_util::device_tuple(
        self.is_device_allocated(), self.is_host_allocated(), "DenseMatrix");
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

template <typename T>
inline void copy_dlpack_to_dense_matrix_typed(PyDenseMatrix&          output,
                                              const dlpack::DLTensor& tensor,
                                              rxmesh::locationT       location,
                                              int64_t                 stride0,
                                              int64_t                 stride1,
                                              cudaStream_t copy_stream)
{
    using namespace rxmesh;

    auto* typed = dynamic_cast<PyDenseMatrixT<T>*>(&output);
    if (!typed || !typed->matrix) {
        throw std::invalid_argument(
            "DenseMatrix.from_dlpack_copy() internal dtype mismatch.");
    }

    auto& mat = *typed->matrix;
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
               copy_stream>>>(mat.data(rxmesh::DEVICE),
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
    py::object capsule = dlpack_util::acquire_capsule(
        std::move(source), "DenseMatrix.from_dlpack_copy()");
    auto* managed = dlpack_util::extract_managed(capsule);
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

        const std::string dtype = dlpack_util::dtype_to_name(tensor.dtype);
        const int64_t     stride0 =
            tensor.strides ? tensor.strides[0] : tensor.shape[1];
        const int64_t stride1 = tensor.strides ? tensor.strides[1] : 1;

        auto output = dispatch_numeric_dtype_str(
            dtype, [&](auto tag) -> std::shared_ptr<PyDenseMatrix> {
                using T  = typename decltype(tag)::type;
                auto out = make_dense_matrix(static_cast<int>(tensor.shape[0]),
                                             static_cast<int>(tensor.shape[1]),
                                             dtype,
                                             static_cast<int>(location),
                                             "col_major");
                const cudaStream_t copy_stream = nullptr;
                copy_dlpack_to_dense_matrix_typed<T>(
                    *out, tensor, location, stride0, stride1, copy_stream);
                return out;
            });
        dlpack_util::mark_consumed(capsule, managed);
        return output;
    } catch (...) {
        dlpack_util::mark_consumed(capsule, managed);
        throw;
    }
}

}  // namespace pyrxmesh_py
