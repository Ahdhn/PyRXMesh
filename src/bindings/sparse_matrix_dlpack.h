#pragma once

#include "bindings/dlpack_minimal.h"
#include "bindings/py_sparse_matrix.h"

#include <cuda_runtime_api.h>
#include <vector>

namespace pyrxmesh_py {

struct SparseDlpackContext
{
    std::shared_ptr<PySparseMatrix> owner;
    int64_t                         shape[1]   = {0};
    int64_t                         strides[1] = {1};
};

inline bool sparse_is_cuda_dlpack_no_sync_stream(py::object stream)
{
    return !stream.is_none() && cuda_stream_arg_value(std::move(stream)) == -1;
}

inline py::object sparse_default_cuda_dlpack_stream_arg()
{
    return py::int_(1);
}

inline bool sparse_source_dlpack_device_is_cuda(py::object source)
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

inline py::object sparse_acquire_dlpack_capsule(py::object source,
                                                py::object stream)
{
    if (PyCapsule_IsValid(source.ptr(), "dltensor")) {
        return source;
    }
    if (!py::hasattr(source, "__dlpack__")) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() expects a DLPack capsule or an "
            "object with __dlpack__().");
    }
    if (sparse_source_dlpack_device_is_cuda(source)) {
        py::dict kwargs;
        kwargs["stream"] =
            stream.is_none() ? sparse_default_cuda_dlpack_stream_arg() : stream;
        return source.attr("__dlpack__")(**kwargs);
    }
    return source.attr("__dlpack__")();
}

inline bool sparse_dlpack_dtype_equal(dlpack::DLDataType a,
                                      dlpack::DLDataType b)
{
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

inline void sparse_synchronize_dlpack_export_stream(py::object stream)
{
    using namespace rxmesh;
    if (sparse_is_cuda_dlpack_no_sync_stream(stream)) {
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

inline void sparse_mark_dlpack_capsule_consumed(
    py::object               capsule,
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
__global__ void sparse_copy_strided_1d_kernel(T*       dst,
                                              const T* src,
                                              int64_t  n,
                                              int64_t  stride)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx * stride];
    }
}

template <typename T>
inline const T* sparse_dlpack_data_ptr(const dlpack::DLTensor& tensor)
{
    return reinterpret_cast<const T*>(static_cast<const char*>(tensor.data) +
                                      tensor.byte_offset);
}

template <typename T>
inline void sparse_copy_dlpack_1d_to_host(const dlpack::DLTensor& tensor,
                                          T*                      dst,
                                          int64_t                 size,
                                          const char*             label,
                                          cudaStream_t            copy_stream)
{
    using namespace rxmesh;

    if (tensor.ndim != 1 || tensor.shape[0] != size) {
        throw std::invalid_argument(std::string(label) +
                                    " must be a 1D tensor with the expected "
                                    "length.");
    }
    const int64_t stride = tensor.strides ? tensor.strides[0] : 1;
    const auto*   src    = sparse_dlpack_data_ptr<T>(tensor);

    if (tensor.device.device_type == dlpack::kDLCPU) {
        for (int64_t i = 0; i < size; ++i) {
            dst[i] = src[i * stride];
        }
        return;
    }
    if (tensor.device.device_type != dlpack::kDLCUDA) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() supports CPU and CUDA DLPack "
            "tensors.");
    }

    if (stride == 1) {
        CUDA_ERROR(cudaMemcpyAsync(dst,
                                   src,
                                   static_cast<size_t>(size) * sizeof(T),
                                   cudaMemcpyDeviceToHost,
                                   copy_stream));
        CUDA_ERROR(cudaStreamSynchronize(copy_stream));
        return;
    }

    T* d_compact = nullptr;
    CUDA_ERROR(cudaMalloc(&d_compact, static_cast<size_t>(size) * sizeof(T)));
    constexpr int threads = 256;
    sparse_copy_strided_1d_kernel<T>
        <<<static_cast<int>((size + threads - 1) / threads),
           threads,
           0,
           copy_stream>>>(d_compact, src, size, stride);
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaMemcpyAsync(dst,
                               d_compact,
                               static_cast<size_t>(size) * sizeof(T),
                               cudaMemcpyDeviceToHost,
                               copy_stream));
    CUDA_ERROR(cudaStreamSynchronize(copy_stream));
    CUDA_ERROR(cudaFree(d_compact));
}

template <typename T>
inline std::shared_ptr<PySparseMatrix> sparse_matrix_from_host_csr_vectors(
    std::vector<typename rxmesh::SparseMatrix<T>::IndexT> row_ptr,
    std::vector<typename rxmesh::SparseMatrix<T>::IndexT> col_idx,
    std::vector<T>                                        values,
    int                                                   rows,
    int                                                   cols)
{
    using namespace rxmesh;

    using IndexT  = typename rxmesh::SparseMatrix<T>::IndexT;
    const int nnz = static_cast<int>(values.size());
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() rows and cols must be positive.");
    }
    if (row_ptr.size() != static_cast<size_t>(rows + 1)) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() row_ptr length must be rows + 1.");
    }
    if (col_idx.size() != values.size()) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() col_idx and values length must "
            "match.");
    }
    if (row_ptr.front() != 0 || row_ptr.back() != nnz) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() row_ptr must start at 0 and end "
            "at nnz.");
    }
    for (int i = 0; i < rows; ++i) {
        if (row_ptr[static_cast<size_t>(i + 1)] <
            row_ptr[static_cast<size_t>(i)]) {
            throw std::invalid_argument(
                "SparseMatrix.from_dlpack_copy() row_ptr must be monotonic.");
        }
    }
    for (int i = 0; i < nnz; ++i) {
        if (col_idx[static_cast<size_t>(i)] < 0 ||
            col_idx[static_cast<size_t>(i)] >= cols) {
            throw std::invalid_argument(
                "SparseMatrix.from_dlpack_copy() column index is out of "
                "range.");
        }
    }

    auto buffers       = std::make_shared<OwnedCsrBuffers<T>>();
    buffers->h_row_ptr = static_cast<IndexT*>(
        malloc(static_cast<size_t>(rows + 1) * sizeof(IndexT)));
    buffers->h_col_idx =
        static_cast<IndexT*>(malloc(static_cast<size_t>(nnz) * sizeof(IndexT)));
    buffers->h_val =
        static_cast<T*>(malloc(static_cast<size_t>(nnz) * sizeof(T)));
    if (!buffers->h_row_ptr || !buffers->h_col_idx || !buffers->h_val) {
        throw std::bad_alloc();
    }

    for (int i = 0; i < rows + 1; ++i) {
        buffers->h_row_ptr[i] = row_ptr[static_cast<size_t>(i)];
    }
    for (int i = 0; i < nnz; ++i) {
        buffers->h_col_idx[i] = col_idx[static_cast<size_t>(i)];
        buffers->h_val[i]     = values[static_cast<size_t>(i)];
    }

    CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&buffers->d_row_ptr),
                          static_cast<size_t>(rows + 1) * sizeof(IndexT)));
    CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&buffers->d_col_idx),
                          static_cast<size_t>(nnz) * sizeof(IndexT)));
    CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&buffers->d_val),
                          static_cast<size_t>(nnz) * sizeof(T)));
    CUDA_ERROR(cudaMemcpy(buffers->d_row_ptr,
                          buffers->h_row_ptr,
                          static_cast<size_t>(rows + 1) * sizeof(IndexT),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(buffers->d_col_idx,
                          buffers->h_col_idx,
                          static_cast<size_t>(nnz) * sizeof(IndexT),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(buffers->d_val,
                          buffers->h_val,
                          static_cast<size_t>(nnz) * sizeof(T),
                          cudaMemcpyHostToDevice));

    auto matrix = std::make_shared<rxmesh::SparseMatrix<T>>(rows,
                                                            cols,
                                                            nnz,
                                                            buffers->d_row_ptr,
                                                            buffers->d_col_idx,
                                                            buffers->d_val,
                                                            buffers->h_row_ptr,
                                                            buffers->h_col_idx,
                                                            buffers->h_val);
    return std::make_shared<PySparseMatrixT<T>>(std::move(matrix),
                                                std::move(buffers),
                                                rxmesh::Op::INVALID,
                                                rxmesh::LOCATION_ALL);
}

inline void sparse_dlpack_capsule_destructor(PyObject* capsule)
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

inline void sparse_dlpack_managed_tensor_deleter(dlpack::DLManagedTensor* self)
{
    delete static_cast<SparseDlpackContext*>(self->manager_ctx);
    delete self;
}

template <typename T>
inline bool fill_sparse_component_dlpack(std::shared_ptr<PySparseMatrix>& owner,
                                         dlpack::DLManagedTensor* managed,
                                         SparseDlpackContext*     context,
                                         CsrComponent             component,
                                         rxmesh::locationT        location)
{
    auto typed = std::dynamic_pointer_cast<PySparseMatrixT<T>>(owner);
    if (!typed) {
        return false;
    }
    using IndexT = typename PySparseMatrixT<T>::IndexT;
    auto& matrix = *typed->matrix;

    if (component == CsrComponent::RowPtr) {
        context->shape[0]        = matrix.rows() + 1;
        context->strides[0]      = 1;
        managed->dl_tensor.data  = matrix.row_ptr(location);
        managed->dl_tensor.dtype = sparse_value_dlpack_dtype<IndexT>();
        return true;
    }
    if (component == CsrComponent::ColIdx) {
        context->shape[0]        = matrix.non_zeros();
        context->strides[0]      = 1;
        managed->dl_tensor.data  = matrix.col_idx(location);
        managed->dl_tensor.dtype = sparse_value_dlpack_dtype<IndexT>();
        return true;
    }

    context->shape[0]        = matrix.non_zeros();
    context->strides[0]      = 1;
    managed->dl_tensor.data  = matrix.val_ptr(location);
    managed->dl_tensor.dtype = sparse_value_dlpack_dtype<T>();
    return true;
}

inline py::capsule sparse_component_to_dlpack(
    std::shared_ptr<PySparseMatrix> self,
    CsrComponent                    component,
    int                             location,
    py::object                      stream)
{
    using namespace rxmesh;
    const auto loc = parse_location(location);
    if (loc != rxmesh::HOST && loc != rxmesh::DEVICE) {
        throw std::invalid_argument(
            "SparseMatrix DLPack location must be Location.HOST or "
            "Location.DEVICE.");
    }
    if (loc == rxmesh::HOST) {
        if (!stream.is_none()) {
            throw std::invalid_argument(
                "SparseMatrix.to_dlpack(Location.HOST) requires stream=None.");
        }
        self->ensure_host_readable();
    } else {
        self->ensure_device_readable();
        sparse_synchronize_dlpack_export_stream(std::move(stream));
    }

    auto* managed  = new dlpack::DLManagedTensor();
    auto* context  = new SparseDlpackContext();
    context->owner = std::move(self);

    int device_id = 0;
    if (loc == rxmesh::DEVICE) {
        CUDA_ERROR(cudaGetDevice(&device_id));
    }

    if (!fill_sparse_component_dlpack<float>(
            context->owner, managed, context, component, loc) &&
        !fill_sparse_component_dlpack<double>(
            context->owner, managed, context, component, loc) &&
        !fill_sparse_component_dlpack<int32_t>(
            context->owner, managed, context, component, loc)) {
        throw std::runtime_error("SparseMatrix DLPack export dtype mismatch.");
    }

    managed->dl_tensor.device = {
        loc == rxmesh::DEVICE ? dlpack::kDLCUDA : dlpack::kDLCPU, device_id};
    managed->dl_tensor.ndim        = 1;
    managed->dl_tensor.shape       = context->shape;
    managed->dl_tensor.strides     = context->strides;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx           = context;
    managed->deleter               = sparse_dlpack_managed_tensor_deleter;

    return py::capsule(managed, "dltensor", sparse_dlpack_capsule_destructor);
}

template <typename T>
inline void sparse_values_from_dlpack_copy_typed(PySparseMatrix&         self,
                                                 const dlpack::DLTensor& tensor,
                                                 rxmesh::locationT       target,
                                                 py::object              stream)
{
    const auto copy_stream = parse_cuda_stream_arg(stream);
    if (!sparse_dlpack_dtype_equal(tensor.dtype,
                                   sparse_value_dlpack_dtype<T>())) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_values_copy() values dtype does not "
            "match the SparseMatrix dtype.");
    }
    if ((target & (rxmesh::HOST | rxmesh::DEVICE)) == rxmesh::LOCATION_NONE) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_values_copy() target must include "
            "Location.HOST or Location.DEVICE.");
    }

    std::vector<T> host(static_cast<size_t>(self.nnz()));
    sparse_copy_dlpack_1d_to_host<T>(
        tensor, host.data(), self.nnz(), "SparseMatrix values", copy_stream);

    auto* typed = dynamic_cast<PySparseMatrixT<T>*>(&self);
    if (!typed) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_values_copy() requires matching values "
            "dtype.");
    }
    self.ensure_host_readable();
    T* dst = typed->matrix->val_ptr(rxmesh::HOST);
    for (int i = 0; i < self.nnz(); ++i) {
        dst[i] = host[static_cast<size_t>(i)];
    }

    if ((target & rxmesh::DEVICE) == rxmesh::DEVICE) {
        self.sync_host_to_device(std::move(stream));
    }
}

inline void sparse_values_from_dlpack_copy(PySparseMatrix& self,
                                           py::object      source,
                                           int             target,
                                           py::object      stream)
{
    py::object capsule =
        sparse_acquire_dlpack_capsule(std::move(source), stream);
    auto* managed = static_cast<dlpack::DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    if (!managed) {
        PyErr_Clear();
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_values_copy() received an invalid "
            "DLPack capsule.");
    }

    try {
        const auto  dst    = parse_location(target);
        const auto& tensor = managed->dl_tensor;
        if (self.dtype() == "float32") {
            sparse_values_from_dlpack_copy_typed<float>(
                self, tensor, dst, stream);
        } else if (self.dtype() == "float64") {
            sparse_values_from_dlpack_copy_typed<double>(
                self, tensor, dst, stream);
        } else if (self.dtype() == "int32") {
            sparse_values_from_dlpack_copy_typed<int32_t>(
                self, tensor, dst, stream);
        }
        sparse_mark_dlpack_capsule_consumed(capsule, managed);
    } catch (...) {
        sparse_mark_dlpack_capsule_consumed(capsule, managed);
        throw;
    }
}

template <typename T>
inline std::shared_ptr<PySparseMatrix> sparse_matrix_from_dlpack_copy_typed(
    const dlpack::DLTensor& row_ptr_tensor,
    const dlpack::DLTensor& col_idx_tensor,
    const dlpack::DLTensor& values_tensor,
    py::tuple               shape,
    cudaStream_t            copy_stream)
{
    using IndexT = typename rxmesh::SparseMatrix<T>::IndexT;

    if (py::len(shape) != 2) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() shape must be a (rows, cols) "
            "tuple.");
    }
    const int rows = shape[0].cast<int>();
    const int cols = shape[1].cast<int>();
    if (!sparse_dlpack_dtype_equal(row_ptr_tensor.dtype,
                                   sparse_value_dlpack_dtype<IndexT>()) ||
        !sparse_dlpack_dtype_equal(col_idx_tensor.dtype,
                                   sparse_value_dlpack_dtype<IndexT>())) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() row_ptr and col_idx must use the "
            "SparseMatrix index dtype.");
    }
    if (!sparse_dlpack_dtype_equal(values_tensor.dtype,
                                   sparse_value_dlpack_dtype<T>())) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() values dtype does not match the "
            "requested SparseMatrix dtype.");
    }
    if (row_ptr_tensor.ndim != 1 || row_ptr_tensor.shape[0] != rows + 1) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() row_ptr length must be rows + 1.");
    }
    if (col_idx_tensor.ndim != 1 || values_tensor.ndim != 1 ||
        col_idx_tensor.shape[0] != values_tensor.shape[0]) {
        throw std::invalid_argument(
            "SparseMatrix.from_dlpack_copy() col_idx and values must be 1D "
            "tensors with matching length.");
    }

    const int           nnz = static_cast<int>(values_tensor.shape[0]);
    std::vector<IndexT> row_ptr(static_cast<size_t>(rows + 1));
    std::vector<IndexT> col_idx(static_cast<size_t>(nnz));
    std::vector<T>      values(static_cast<size_t>(nnz));

    sparse_copy_dlpack_1d_to_host<IndexT>(row_ptr_tensor,
                                          row_ptr.data(),
                                          rows + 1,
                                          "SparseMatrix row_ptr",
                                          copy_stream);
    sparse_copy_dlpack_1d_to_host<IndexT>(col_idx_tensor,
                                          col_idx.data(),
                                          nnz,
                                          "SparseMatrix col_idx",
                                          copy_stream);
    sparse_copy_dlpack_1d_to_host<T>(
        values_tensor, values.data(), nnz, "SparseMatrix values", copy_stream);

    return sparse_matrix_from_host_csr_vectors<T>(
        std::move(row_ptr), std::move(col_idx), std::move(values), rows, cols);
}

inline std::shared_ptr<PySparseMatrix> sparse_matrix_from_dlpack_copy(
    py::object  row_ptr,
    py::object  col_idx,
    py::object  values,
    py::tuple   shape,
    std::string dtype,
    py::object  stream)
{
    const auto copy_stream = parse_cuda_stream_arg(stream);
    py::object row_capsule =
        sparse_acquire_dlpack_capsule(std::move(row_ptr), stream);
    py::object col_capsule =
        sparse_acquire_dlpack_capsule(std::move(col_idx), stream);
    py::object val_capsule =
        sparse_acquire_dlpack_capsule(std::move(values), stream);

    dlpack::DLManagedTensor* row_managed = nullptr;
    dlpack::DLManagedTensor* col_managed = nullptr;
    dlpack::DLManagedTensor* val_managed = nullptr;

    auto consume = [&]() {
        if (row_managed) {
            sparse_mark_dlpack_capsule_consumed(row_capsule, row_managed);
        }
        if (col_managed) {
            sparse_mark_dlpack_capsule_consumed(col_capsule, col_managed);
        }
        if (val_managed) {
            sparse_mark_dlpack_capsule_consumed(val_capsule, val_managed);
        }
    };

    try {
        row_managed = static_cast<dlpack::DLManagedTensor*>(
            PyCapsule_GetPointer(row_capsule.ptr(), "dltensor"));
        col_managed = static_cast<dlpack::DLManagedTensor*>(
            PyCapsule_GetPointer(col_capsule.ptr(), "dltensor"));
        val_managed = static_cast<dlpack::DLManagedTensor*>(
            PyCapsule_GetPointer(val_capsule.ptr(), "dltensor"));
        if (!row_managed || !col_managed || !val_managed) {
            PyErr_Clear();
            throw std::invalid_argument(
                "SparseMatrix.from_dlpack_copy() received an invalid DLPack "
                "capsule.");
        }

        if (dtype.empty()) {
            if (sparse_dlpack_dtype_equal(val_managed->dl_tensor.dtype,
                                          sparse_value_dlpack_dtype<float>())) {
                dtype = "float32";
            } else if (sparse_dlpack_dtype_equal(
                           val_managed->dl_tensor.dtype,
                           sparse_value_dlpack_dtype<double>())) {
                dtype = "float64";
            } else if (sparse_dlpack_dtype_equal(
                           val_managed->dl_tensor.dtype,
                           sparse_value_dlpack_dtype<int32_t>())) {
                dtype = "int32";
            }
        }

        std::shared_ptr<PySparseMatrix> result;
        switch (parse_dtype(dtype)) {
            case DType::Float32:
                result = sparse_matrix_from_dlpack_copy_typed<float>(
                    row_managed->dl_tensor,
                    col_managed->dl_tensor,
                    val_managed->dl_tensor,
                    shape,
                    copy_stream);
                break;
            case DType::Float64:
                result = sparse_matrix_from_dlpack_copy_typed<double>(
                    row_managed->dl_tensor,
                    col_managed->dl_tensor,
                    val_managed->dl_tensor,
                    shape,
                    copy_stream);
                break;
            case DType::Int32:
                result = sparse_matrix_from_dlpack_copy_typed<int32_t>(
                    row_managed->dl_tensor,
                    col_managed->dl_tensor,
                    val_managed->dl_tensor,
                    shape,
                    copy_stream);
                break;
            default:
                throw std::invalid_argument(
                    "SparseMatrix.from_dlpack_copy() supports float32, "
                    "float64, and int32 values.");
        }
        consume();
        return result;
    } catch (...) {
        consume();
        throw;
    }
}

}  // namespace pyrxmesh_py
