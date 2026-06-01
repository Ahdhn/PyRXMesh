#pragma once

#include "bindings/py_sparse_matrix.h"

#include <numeric>
#include <unordered_map>

namespace pyrxmesh_py {

template <typename T>
inline std::shared_ptr<PySparseMatrix> sparse_matrix_from_csr_typed(
    py::array row_ptr,
    py::array col_idx,
    py::array values,
    py::tuple shape)
{
    using IndexT = typename rxmesh::SparseMatrix<T>::IndexT;

    if (py::len(shape) != 2) {
        throw std::invalid_argument(
            "SparseMatrix.from_numpy_copy() shape must be a (rows, cols) "
            "tuple.");
    }
    const int rows = shape[0].cast<int>();
    const int cols = shape[1].cast<int>();
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "SparseMatrix.from_numpy_copy() rows and cols must be positive.");
    }

    py::array_t<IndexT, py::array::c_style | py::array::forcecast> rp(row_ptr);
    py::array_t<IndexT, py::array::c_style | py::array::forcecast> ci(col_idx);
    py::array_t<T, py::array::c_style | py::array::forcecast>      val(values);

    const py::buffer_info rp_info  = rp.request();
    const py::buffer_info ci_info  = ci.request();
    const py::buffer_info val_info = val.request();
    if (rp_info.ndim != 1 || rp_info.shape[0] != rows + 1) {
        throw std::invalid_argument(
            "SparseMatrix.from_numpy_copy() row_ptr length must be rows + 1.");
    }
    if (ci_info.ndim != 1 || val_info.ndim != 1 ||
        ci_info.shape[0] != val_info.shape[0]) {
        throw std::invalid_argument(
            "SparseMatrix.from_numpy_copy() col_idx and values must be 1D "
            "arrays with matching length.");
    }

    const int  nnz      = static_cast<int>(ci_info.shape[0]);
    const auto rp_view  = rp.template unchecked<1>();
    const auto ci_view  = ci.template unchecked<1>();
    const auto val_view = val.template unchecked<1>();
    if (rp_view(0) != 0 || rp_view(rows) != nnz) {
        throw std::invalid_argument(
            "SparseMatrix.from_numpy_copy() row_ptr must start at 0 and end at "
            "nnz.");
    }
    for (int i = 0; i < rows; ++i) {
        if (rp_view(i + 1) < rp_view(i)) {
            throw std::invalid_argument(
                "SparseMatrix.from_numpy_copy() row_ptr must be monotonic.");
        }
    }
    for (int i = 0; i < nnz; ++i) {
        if (ci_view(i) < 0 || ci_view(i) >= cols) {
            throw std::invalid_argument(
                "SparseMatrix.from_numpy_copy() column index is out of range.");
        }
    }

    auto buffers = std::make_shared<OwnedCsrBuffers<T>>();

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
        buffers->h_row_ptr[i] = rp_view(i);
    }
    for (int i = 0; i < nnz; ++i) {
        buffers->h_col_idx[i] = ci_view(i);
        buffers->h_val[i]     = val_view(i);
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

    auto                matrix = std::make_shared<rxmesh::SparseMatrix<T>>(rows,
                                                            cols,
                                                            nnz,
                                                            buffers->d_row_ptr,
                                                            buffers->d_col_idx,
                                                            buffers->d_val,
                                                            buffers->h_row_ptr,
                                                            buffers->h_col_idx,
                                                            buffers->h_val);
    SparseMatrixVariant wrapped = matrix;
    OwnedCsrVariant     owned   = buffers;
    return std::make_shared<PySparseMatrix>(std::move(wrapped),
                                            std::move(owned),
                                            rxmesh::Op::INVALID,
                                            rxmesh::LOCATION_ALL);
}

inline std::shared_ptr<PySparseMatrix> sparse_matrix_from_numpy_copy(
    py::array   row_ptr,
    py::array   col_idx,
    py::array   values,
    py::tuple   shape,
    std::string dtype)
{
    switch (parse_dtype(dtype)) {
        case DType::Float32:
            return sparse_matrix_from_csr_typed<float>(
                row_ptr, col_idx, values, shape);
        case DType::Float64:
            return sparse_matrix_from_csr_typed<double>(
                row_ptr, col_idx, values, shape);
        case DType::Int32:
            return sparse_matrix_from_csr_typed<int32_t>(
                row_ptr, col_idx, values, shape);
        default:
            throw std::invalid_argument(
                "SparseMatrix.from_numpy_copy() supports float32, float64, "
                "and int32 values.");
    }
}

}  // namespace pyrxmesh_py
