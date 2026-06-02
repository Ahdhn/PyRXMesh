#pragma once

#include "bindings/common.h"
#include "bindings/dlpack_minimal.h"
#include "bindings/py_dense_matrix.h"

#include <cstdlib>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include <rxmesh/matrix/sparse_matrix.h>

namespace pyrxmesh_py {

enum class CsrComponent
{
    RowPtr,
    ColIdx,
    Values
};

template <typename T>
inline std::string sparse_dtype_name()
{
    return dense_dtype_name<T>();
}

template <typename T>
inline dlpack::DLDataType sparse_value_dlpack_dtype()
{
    if constexpr (std::is_same_v<T, float>) {
        return {2, 32, 1};
    } else if constexpr (std::is_same_v<T, double>) {
        return {2, 64, 1};
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return {0, 32, 1};
    } else {
        static_assert(always_false<T>::value, "Unsupported sparse dtype");
    }
}

template <typename T>
struct OwnedCsrBuffers
{
    using IndexT = typename rxmesh::SparseMatrix<T>::IndexT;

    IndexT* h_row_ptr = nullptr;
    IndexT* h_col_idx = nullptr;
    T*      h_val     = nullptr;
    IndexT* d_row_ptr = nullptr;
    IndexT* d_col_idx = nullptr;
    T*      d_val     = nullptr;

    ~OwnedCsrBuffers()
    {
        if (h_row_ptr) {
            free(h_row_ptr);
        }
        if (h_col_idx) {
            free(h_col_idx);
        }
        if (h_val) {
            free(h_val);
        }
        if (d_row_ptr) {
            cudaFree(d_row_ptr);
        }
        if (d_col_idx) {
            cudaFree(d_col_idx);
        }
        if (d_val) {
            cudaFree(d_val);
        }
    }
};

struct PySparseMatrix : std::enable_shared_from_this<PySparseMatrix>
{
    std::shared_ptr<rxmesh::RXMeshStatic> mesh_owner;
    rxmesh::Op                            op        = rxmesh::Op::INVALID;
    rxmesh::locationT                     allocated = rxmesh::LOCATION_NONE;
    bool                                  released  = false;

    virtual ~PySparseMatrix() = default;

    virtual int         rows() const                                        = 0;
    virtual int         cols() const                                        = 0;
    virtual int         nnz() const                                         = 0;
    virtual std::string dtype() const                                       = 0;
    virtual int         lower_nnz() const                                   = 0;
    virtual void        release()                                           = 0;
    virtual void        reset(py::object, int, py::object)                  = 0;
    virtual void        move(int, int, py::object)                          = 0;
    virtual void        copy_from(PySparseMatrix&, int, int, py::object)    = 0;
    virtual bool        is_non_zero(int, int) const                         = 0;
    virtual py::object  value(int, int) const                               = 0;
    virtual void        set_value(int, int, py::object)                     = 0;
    virtual py::tuple   to_numpy(int)                                       = 0;
    virtual py::tuple   to_numpy_copy(int)                                  = 0;
    virtual py::array   values_to_numpy(int)                                = 0;
    virtual py::array   values_to_numpy_copy(int)                           = 0;
    virtual void        from_numpy_values_copy(py::array, int, py::object)  = 0;
    virtual std::shared_ptr<PyDenseMatrix> multiply_dense(PyDenseMatrix&,
                                                          bool,
                                                          bool,
                                                          py::object,
                                                          py::object,
                                                          py::object)       = 0;
    virtual std::shared_ptr<PyDenseMatrix> multiply_vector(PyDenseMatrix&,
                                                           py::object)      = 0;
    virtual void                           to_mtx(const std::string&) const = 0;
    virtual void to_file(const std::string&) const                          = 0;

    std::pair<int, int> shape() const
    {
        return {rows(), cols()};
    }

    std::string index_dtype() const
    {
        return "int32";
    }

    int location() const
    {
        return static_cast<int>(allocated);
    }

    bool is_host_allocated() const
    {
        return (allocated & rxmesh::HOST) == rxmesh::HOST;
    }

    bool is_device_allocated() const
    {
        return (allocated & rxmesh::DEVICE) == rxmesh::DEVICE;
    }

    void ensure_not_released() const
    {
        if (released) {
            throw std::runtime_error("SparseMatrix has been released.");
        }
    }

    void ensure_host_readable()
    {
        ensure_not_released();
        if (!is_host_allocated()) {
            if (!is_device_allocated()) {
                throw std::runtime_error(
                    "SparseMatrix HOST memory is not allocated.");
            }
            move(static_cast<int>(rxmesh::DEVICE),
                 static_cast<int>(rxmesh::HOST),
                 py::none());
        }
    }

    void ensure_device_readable(py::object stream)
    {
        ensure_not_released();
        if (!is_device_allocated()) {
            if (!is_host_allocated()) {
                throw std::runtime_error(
                    "SparseMatrix DEVICE memory is not allocated.");
            }
            move(static_cast<int>(rxmesh::HOST),
                 static_cast<int>(rxmesh::DEVICE),
                 std::move(stream));
        }
    }

    void ensure_device_readable()
    {
        ensure_device_readable(py::none());
    }

    void sync_host_to_device(py::object stream)
    {
        if (is_device_allocated() && is_host_allocated()) {
            move(static_cast<int>(rxmesh::HOST),
                 static_cast<int>(rxmesh::DEVICE),
                 std::move(stream));
        }
    }

    void sync_device_to_host(py::object stream)
    {
        if (is_device_allocated() && is_host_allocated()) {
            move(static_cast<int>(rxmesh::DEVICE),
                 static_cast<int>(rxmesh::HOST),
                 std::move(stream));
        }
    }
};

template <typename T>
struct PySparseMatrixT : PySparseMatrix
{
    using SpMatT = rxmesh::SparseMatrix<T>;
    using IndexT = typename SpMatT::IndexT;

    std::shared_ptr<SpMatT>             matrix;
    std::shared_ptr<OwnedCsrBuffers<T>> owned_buffers;

    PySparseMatrixT()
    {
        released = true;
    }

    PySparseMatrixT(std::shared_ptr<rxmesh::RXMeshStatic> owner,
                    rxmesh::Op                            op_in,
                    int                                   location)
    {
        if (!owner) {
            throw std::invalid_argument(
                "SparseMatrix requires a live RXMeshStatic owner.");
        }
        mesh_owner     = std::move(owner);
        op             = op_in;
        const auto loc = parse_location(location);
        matrix         = std::make_shared<SpMatT>(*mesh_owner, op);
        allocated      = loc;
        released       = false;
    }

    PySparseMatrixT(std::shared_ptr<SpMatT>               in_matrix,
                    std::shared_ptr<OwnedCsrBuffers<T>>   buffers,
                    rxmesh::Op                            op_in,
                    rxmesh::locationT                     location,
                    std::shared_ptr<rxmesh::RXMeshStatic> owner = nullptr)
    {
        matrix        = std::move(in_matrix);
        owned_buffers = std::move(buffers);
        mesh_owner    = std::move(owner);
        op            = op_in;
        allocated     = location;
        released      = false;
    }

    ~PySparseMatrixT() override
    {
        release();
    }

    int rows() const override
    {
        return static_cast<int>(matrix->rows());
    }

    int cols() const override
    {
        return static_cast<int>(matrix->cols());
    }

    int nnz() const override
    {
        return static_cast<int>(matrix->non_zeros());
    }

    std::string dtype() const override
    {
        return sparse_dtype_name<T>();
    }

    int lower_nnz() const override
    {
        const_cast<PySparseMatrixT*>(this)->ensure_host_readable();
        return static_cast<int>(matrix->lower_non_zeros());
    }

    void release() override
    {
        if (!released && matrix) {
            matrix->release();
            released  = true;
            allocated = rxmesh::LOCATION_NONE;
        }
    }

    void reset(py::object value, int location, py::object stream) override
    {
        ensure_not_released();
        matrix->reset(value.cast<T>(),
                      parse_location(location),
                      parse_cuda_stream_arg(stream));
    }

    void move(int source, int target, py::object stream) override
    {
        ensure_not_released();
        const auto   src       = parse_location(source);
        const auto   dst       = parse_location(target);
        cudaStream_t cuda_strm = parse_cuda_stream_arg(stream);
        matrix->move(src, dst, cuda_strm);
        allocated = static_cast<rxmesh::locationT>(static_cast<int>(allocated) |
                                                   static_cast<int>(dst));
    }

    void copy_from(PySparseMatrix& other,
                   int             source,
                   int             target,
                   py::object      stream) override
    {
        ensure_not_released();
        other.ensure_not_released();
        auto* typed_other = dynamic_cast<PySparseMatrixT<T>*>(&other);
        if (!typed_other) {
            throw std::invalid_argument(
                "SparseMatrix.copy_from() requires matching dtypes.");
        }
        if (typed_other->rows() != rows() || typed_other->cols() != cols() ||
            typed_other->nnz() != nnz()) {
            throw std::invalid_argument(
                "SparseMatrix.copy_from() requires matching shape and nnz.");
        }

        const auto src = parse_location(source);
        const auto dst = parse_location(target);
        if ((src & rxmesh::HOST) == rxmesh::HOST) {
            typed_other->ensure_host_readable();
        }
        if ((src & rxmesh::DEVICE) == rxmesh::DEVICE) {
            typed_other->ensure_device_readable(stream);
        }
        matrix->copy_from(
            *typed_other->matrix, src, dst, parse_cuda_stream_arg(stream));
        allocated = static_cast<rxmesh::locationT>(static_cast<int>(allocated) |
                                                   static_cast<int>(dst));
    }

    bool is_non_zero(int row, int col) const override
    {
        const_cast<PySparseMatrixT*>(this)->ensure_host_readable();
        return matrix->is_non_zero(row, col);
    }

    py::object value(int row, int col) const override
    {
        const_cast<PySparseMatrixT*>(this)->ensure_host_readable();
        return py::cast((*matrix)(row, col));
    }

    void set_value(int row, int col, py::object value) override
    {
        ensure_host_readable();
        (*matrix)(row, col) = value.cast<T>();
        sync_host_to_device(py::none());
    }

    py::array row_ptr_to_numpy_impl(int source, bool copy)
    {
        const auto src = parse_location(source);
        if (src != rxmesh::HOST) {
            throw std::invalid_argument(
                "SparseMatrix.to_numpy() only supports Location.HOST.");
        }
        ensure_host_readable();
        return make_numpy_1d<IndexT>(matrix->row_ptr(src), rows() + 1, copy);
    }

    py::array col_indices_to_numpy_impl(int source, bool copy)
    {
        const auto src = parse_location(source);
        if (src != rxmesh::HOST) {
            throw std::invalid_argument(
                "SparseMatrix.to_numpy() only supports Location.HOST.");
        }
        ensure_host_readable();
        return make_numpy_1d<IndexT>(matrix->col_idx(src), nnz(), copy);
    }

    py::array values_to_numpy_impl(int source, bool copy)
    {
        const auto src = parse_location(source);
        if (src != rxmesh::HOST) {
            throw std::invalid_argument(
                "SparseMatrix.to_numpy() only supports Location.HOST.");
        }
        ensure_host_readable();
        return make_numpy_1d<T>(matrix->val_ptr(src), nnz(), copy);
    }

    py::tuple to_numpy(int location) override
    {
        return py::make_tuple(row_ptr_to_numpy_impl(location, false),
                              col_indices_to_numpy_impl(location, false),
                              values_to_numpy_impl(location, false));
    }

    py::tuple to_numpy_copy(int source) override
    {
        return py::make_tuple(row_ptr_to_numpy_impl(source, true),
                              col_indices_to_numpy_impl(source, true),
                              values_to_numpy_impl(source, true));
    }

    py::array values_to_numpy(int location) override
    {
        return values_to_numpy_impl(location, false);
    }

    py::array values_to_numpy_copy(int source) override
    {
        return values_to_numpy_impl(source, true);
    }

    void from_numpy_values_copy(py::array  values,
                                int        target,
                                py::object stream) override
    {
        py::array_t<T, py::array::c_style | py::array::forcecast> typed(values);
        const auto info = typed.request();
        if (info.ndim != 1 || info.shape[0] != nnz()) {
            throw std::invalid_argument(
                "SparseMatrix.from_numpy_values_copy() values must be a 1D "
                "array with length nnz.");
        }

        ensure_host_readable();
        std::memcpy(matrix->val_ptr(rxmesh::HOST),
                    info.ptr,
                    static_cast<size_t>(nnz()) * sizeof(T));

        const auto dst = parse_location(target);
        if ((dst & rxmesh::DEVICE) == rxmesh::DEVICE) {
            move(static_cast<int>(rxmesh::HOST),
                 static_cast<int>(rxmesh::DEVICE),
                 std::move(stream));
        }
    }

    std::shared_ptr<PyDenseMatrix> multiply_dense(PyDenseMatrix& rhs,
                                                  bool           transpose_a,
                                                  bool           transpose_b,
                                                  py::object     alpha,
                                                  py::object     beta,
                                                  py::object stream) override
    {
        ensure_device_readable(stream);
        if (rhs.dtype() != dtype()) {
            throw std::invalid_argument(
                "SparseMatrix.multiply() requires matching sparse and dense "
                "dtypes.");
        }

        const int out_rows = transpose_a ? cols() : rows();
        const int out_cols = transpose_b ? rhs.rows() : rhs.cols();
        auto      output   = std::make_shared<PyDenseMatrix>(
            dtype(),
            out_rows,
            out_cols,
            static_cast<int>(rxmesh::LOCATION_ALL),
            "col_major");
        const auto cuda_stream = parse_cuda_stream_arg(stream);
        std::visit(
            [&](const auto& rhs_mat, const auto& out_mat) {
                using RhsMatT = std::decay_t<decltype(rhs_mat)>;
                using OutMatT = std::decay_t<decltype(out_mat)>;
                if constexpr (
                    std::is_same_v<T, typename RhsMatT::element_type::Type> &&
                    std::is_same_v<T, typename OutMatT::element_type::Type>) {
                    matrix->multiply(*rhs_mat,
                                     *out_mat,
                                     transpose_a,
                                     transpose_b,
                                     alpha.cast<T>(),
                                     beta.cast<T>(),
                                     cuda_stream);
                } else {
                    throw std::invalid_argument(
                        "SparseMatrix.multiply() requires matching dtypes.");
                }
            },
            rhs.matrix,
            output->matrix);

        output->move(rxmesh::DEVICE, rxmesh::HOST, cuda_stream);
        return output;
    }

    std::shared_ptr<PyDenseMatrix> multiply_vector(PyDenseMatrix& rhs,
                                                   py::object stream) override
    {
        if (rhs.cols() != 1) {
            throw std::invalid_argument(
                "SparseMatrix.multiply_vector() requires a DenseMatrix with "
                "one column.");
        }
        ensure_device_readable(stream);
        if (rhs.dtype() != dtype()) {
            throw std::invalid_argument(
                "SparseMatrix.multiply_vector() requires matching sparse and "
                "dense dtypes.");
        }

        auto output = std::make_shared<PyDenseMatrix>(
            dtype(),
            rows(),
            1,
            static_cast<int>(rxmesh::LOCATION_ALL),
            "col_major");
        const auto cuda_stream = parse_cuda_stream_arg(stream);
        std::visit(
            [&](const auto& rhs_mat, const auto& out_mat) {
                using RhsMatT = std::decay_t<decltype(rhs_mat)>;
                using OutMatT = std::decay_t<decltype(out_mat)>;
                if constexpr (
                    std::is_same_v<T, typename RhsMatT::element_type::Type> &&
                    std::is_same_v<T, typename OutMatT::element_type::Type>) {
                    matrix->multiply_cw(*rhs_mat, *out_mat, cuda_stream);
                } else {
                    throw std::invalid_argument(
                        "SparseMatrix.multiply_vector() requires matching "
                        "dtypes.");
                }
            },
            rhs.matrix,
            output->matrix);

        output->move(rxmesh::DEVICE, rxmesh::HOST, cuda_stream);
        return output;
    }

    void to_mtx(const std::string& filename) const override
    {
        const_cast<PySparseMatrixT*>(this)->ensure_host_readable();
        matrix->to_mtx(filename);
    }

    void to_file(const std::string& filename) const override
    {
        const_cast<PySparseMatrixT*>(this)->ensure_host_readable();
        matrix->to_file(filename);
    }

   protected:
    template <typename ArrayT>
    py::array make_numpy_1d(ArrayT* ptr, int length, bool copy)
    {
        if (copy) {
            py::array_t<ArrayT> arr({length});
            auto                out = arr.template mutable_unchecked<1>();
            for (int i = 0; i < length; ++i) {
                out(i) = ptr[i];
            }
            return arr;
        }
        return py::array_t<ArrayT>({length},
                                   {static_cast<py::ssize_t>(sizeof(ArrayT))},
                                   ptr,
                                   py::cast(shared_from_this()));
    }
};

template <typename Fn>
void with_float_double_sparse_matrix(PySparseMatrix& matrix,
                                     const char*     api_name,
                                     Fn&&            fn)
{
    if (auto* typed = dynamic_cast<PySparseMatrixT<float>*>(&matrix)) {
        fn(typed->matrix);
        return;
    }
    if (auto* typed = dynamic_cast<PySparseMatrixT<double>*>(&matrix)) {
        fn(typed->matrix);
        return;
    }
    if (dynamic_cast<PySparseMatrixT<int32_t>*>(&matrix)) {
        throw std::invalid_argument(std::string(api_name) +
                                    " support float32 and float64 "
                                    "SparseMatrix values.");
    }
    throw std::runtime_error(std::string(api_name) +
                             " received an unknown SparseMatrix dtype.");
}

}  // namespace pyrxmesh_py
