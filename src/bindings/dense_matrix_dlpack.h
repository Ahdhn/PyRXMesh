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

struct DenseMatrixDlpackViewOwner
{
    dlpack::DLManagedTensor* managed = nullptr;

    explicit DenseMatrixDlpackViewOwner(dlpack::DLManagedTensor* in_managed)
        : managed(in_managed)
    {
    }

    ~DenseMatrixDlpackViewOwner()
    {
        if (managed && managed->deleter) {
            managed->deleter(managed);
        }
    }
};

template <typename T, int Order>
__global__ void copy_dlpack_to_dense_kernel(T*       dst,
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

    int64_t row;
    int64_t col;
    if constexpr (Order == Eigen::RowMajor) {
        row = idx / cols;
        col = idx % cols;
    } else {
        row = idx % rows;
        col = idx / rows;
    }
    dst[idx] = src[row * stride0 + col * stride1];
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

    int device_id = 0;
    if (loc == rxmesh::HOST) {
        if (!stream.is_none()) {
            throw std::invalid_argument(
                "DenseMatrix.to_dlpack(Location.HOST) requires stream=None.");
        }
    } else {
        CUDA_ERROR(cudaGetDevice(&device_id));
        dlpack_util::synchronize_export_stream(std::move(stream));
    }

    auto* managed  = new dlpack::DLManagedTensor();
    auto* context  = new DlpackContext();
    context->owner = std::move(self);

    with_typed_dense_matrix(*context->owner, [&](auto& typed) {
        using TypedT = std::decay_t<decltype(typed)>;
        using T      = typename TypedT::MatT::Type;
        auto& mat    = *typed.matrix;

        context->shape[0] = mat.rows();
        context->shape[1] = mat.cols();
        if constexpr (TypedT::MatrixOrder == Eigen::RowMajor) {
            context->strides[0] = mat.cols();
            context->strides[1] = 1;
        } else {
            context->strides[0] = 1;
            context->strides[1] = mat.rows();
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

template <typename T, int Order>
inline void copy_dlpack_to_dense_matrix_typed(PyDenseMatrix&          output,
                                              const dlpack::DLTensor& tensor,
                                              rxmesh::locationT       location,
                                              int64_t                 stride0,
                                              int64_t                 stride1,
                                              cudaStream_t copy_stream)
{
    using namespace rxmesh;

    auto* typed = dynamic_cast<PyDenseMatrixT<T, Order>*>(&output);
    if (!typed || !typed->matrix) {
        throw std::invalid_argument(
            "DenseMatrix.from_dlpack_copy() internal dtype/order mismatch.");
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
        copy_dlpack_to_dense_kernel<T, Order>
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
    py::object         source,
    const std::string& order = "col_major")
{
    const int  storage_order = parse_dense_matrix_order(order);
    py::object capsule       = dlpack_util::acquire_capsule(
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
                                             order);
                const cudaStream_t copy_stream = nullptr;
                if (storage_order == Eigen::RowMajor) {
                    copy_dlpack_to_dense_matrix_typed<T, Eigen::RowMajor>(
                        *out, tensor, location, stride0, stride1, copy_stream);
                } else {
                    copy_dlpack_to_dense_matrix_typed<T, Eigen::ColMajor>(
                        *out, tensor, location, stride0, stride1, copy_stream);
                }
                return out;
            });
        dlpack_util::mark_consumed(capsule, managed);
        return output;
    } catch (...) {
        dlpack_util::mark_consumed(capsule, managed);
        throw;
    }
}

inline void validate_dense_matrix_view_strides(const dlpack::DLTensor& tensor,
                                               int storage_order)
{
    const int64_t rows = tensor.shape[0];
    const int64_t cols = tensor.shape[1];
    const int64_t stride0 =
        tensor.strides ? tensor.strides[0] : tensor.shape[1];
    const int64_t stride1 = tensor.strides ? tensor.strides[1] : 1;

    if (storage_order == Eigen::RowMajor) {
        if (stride0 != cols || stride1 != 1) {
            throw std::invalid_argument(
                "DenseMatrix.from_dlpack_view() requires compact row_major "
                "strides (cols, 1).");
        }
    } else if (stride0 != 1 || stride1 != rows) {
        throw std::invalid_argument(
            "DenseMatrix.from_dlpack_view() requires compact col_major "
            "strides (1, rows).");
    }
}

template <typename T, int Order>
inline std::shared_ptr<PyDenseMatrix> make_dense_matrix_view_from_dlpack_typed(
    const dlpack::DLTensor& tensor,
    rxmesh::locationT       location,
    std::shared_ptr<void>   owner)
{
    auto* ptr   = reinterpret_cast<T*>(static_cast<char*>(tensor.data) +
                                     tensor.byte_offset);
    T*    h_ptr = location == rxmesh::HOST ? ptr : nullptr;
    T*    d_ptr = location == rxmesh::DEVICE ? ptr : nullptr;

    using MatT = typename PyDenseMatrixT<T, Order>::MatT;
    auto out   = std::make_shared<PyDenseMatrixT<T, Order>>(
        std::make_shared<MatT>(static_cast<int>(tensor.shape[0]),
                               static_cast<int>(tensor.shape[1]),
                               d_ptr,
                               h_ptr),
        location);
    out->external_view  = true;
    out->external_owner = std::move(owner);
    return out;
}

inline void consume_dlpack_capsule_for_view(py::object               capsule,
                                            dlpack::DLManagedTensor* managed)
{
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0 ||
        PyCapsule_SetDestructor(capsule.ptr(), nullptr) != 0) {
        PyErr_Clear();
        dlpack_util::mark_consumed(std::move(capsule), managed);
        throw std::runtime_error(
            "DenseMatrix.from_dlpack_view() failed to consume the DLPack "
            "capsule.");
    }
}

inline std::shared_ptr<PyDenseMatrix> dense_matrix_from_dlpack_view(
    py::object         source,
    const std::string& order = "col_major")
{
    const int  storage_order = parse_dense_matrix_order(order);
    py::object capsule       = dlpack_util::acquire_capsule(
        std::move(source), "DenseMatrix.from_dlpack_view()");
    auto* managed = dlpack_util::extract_managed(capsule);
    if (!managed) {
        PyErr_Clear();
        throw std::invalid_argument(
            "DenseMatrix.from_dlpack_view() received an invalid DLPack "
            "capsule.");
    }

    bool capsule_consumed = false;
    try {
        const dlpack::DLTensor& tensor = managed->dl_tensor;
        if (tensor.ndim != 2) {
            throw std::invalid_argument(
                "DenseMatrix.from_dlpack_view() expects a 2D tensor.");
        }
        if (tensor.shape[0] <= 0 || tensor.shape[1] <= 0) {
            throw std::invalid_argument(
                "DenseMatrix.from_dlpack_view() expects positive tensor "
                "dimensions.");
        }
        validate_dense_matrix_view_strides(tensor, storage_order);

        rxmesh::locationT location;
        if (tensor.device.device_type == dlpack::kDLCPU) {
            location = rxmesh::HOST;
        } else if (tensor.device.device_type == dlpack::kDLCUDA) {
            location = rxmesh::DEVICE;
        } else {
            throw std::invalid_argument(
                "DenseMatrix.from_dlpack_view() supports CPU and CUDA DLPack "
                "tensors.");
        }

        const std::string dtype = dlpack_util::dtype_to_name(tensor.dtype);
        consume_dlpack_capsule_for_view(capsule, managed);
        capsule_consumed = true;
        auto owner = std::make_shared<DenseMatrixDlpackViewOwner>(managed);

        return dispatch_numeric_dtype_str(
            dtype, [&](auto tag) -> std::shared_ptr<PyDenseMatrix> {
                using T = typename decltype(tag)::type;
                if (storage_order == Eigen::RowMajor) {
                    return make_dense_matrix_view_from_dlpack_typed<
                        T,
                        Eigen::RowMajor>(tensor, location, owner);
                }
                return make_dense_matrix_view_from_dlpack_typed<
                    T,
                    Eigen::ColMajor>(tensor, location, owner);
            });
    } catch (...) {
        if (!capsule_consumed) {
            dlpack_util::mark_consumed(capsule, managed);
        }
        throw;
    }
}

}  // namespace pyrxmesh_py
