#pragma once

#include "bindings/dlpack_minimal.h"
#include "bindings/py_dense_matrix.h"

#include <cstdlib>
#include <numeric>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include "rxmesh/matrix/sparse_matrix.h"

namespace pyrxmesh_py {

using SparseMatrixVariant =
    std::variant<std::shared_ptr<rxmesh::SparseMatrix<float>>,
                 std::shared_ptr<rxmesh::SparseMatrix<double>>,
                 std::shared_ptr<rxmesh::SparseMatrix<int32_t>>>;

template <typename T>
struct OwnedCsrBuffers
{
    using IndexT = typename rxmesh::SparseMatrix<T>::IndexT;

    ~OwnedCsrBuffers()
    {
        free(h_row_ptr);
        free(h_col_idx);
        free(h_val);
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

    IndexT* h_row_ptr = nullptr;
    IndexT* h_col_idx = nullptr;
    T*      h_val     = nullptr;
    IndexT* d_row_ptr = nullptr;
    IndexT* d_col_idx = nullptr;
    T*      d_val     = nullptr;
};

using OwnedCsrVariant = std::variant<std::monostate,
                                     std::shared_ptr<OwnedCsrBuffers<float>>,
                                     std::shared_ptr<OwnedCsrBuffers<double>>,
                                     std::shared_ptr<OwnedCsrBuffers<int32_t>>>;

enum class CsrComponent
{
    RowPtr,
    ColIdx,
    Values,
};

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
        static_assert(always_false<T>::value, "Unsupported SparseMatrix dtype");
    }
}

template <typename T>
inline const char* sparse_dtype_name()
{
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    } else {
        static_assert(always_false<T>::value, "Unsupported SparseMatrix dtype");
    }
}

struct PySparseMatrix;

struct SparseDlpackContext
{
    std::shared_ptr<PySparseMatrix> owner;
    int64_t                         shape[1];
    int64_t                         strides[1];
};

struct PySparseMatrix : std::enable_shared_from_this<PySparseMatrix>
{
    PySparseMatrix()
        : op(rxmesh::Op::INVALID),
          allocated(rxmesh::LOCATION_NONE),
          released(true)
    {
    }

    PySparseMatrix(std::shared_ptr<rxmesh::RXMeshStatic> owner,
                   rxmesh::Op                            query_op,
                   std::string                           dtype)
        : mesh_owner(std::move(owner)),
          op(query_op),
          allocated(rxmesh::LOCATION_ALL)
    {
        if (!mesh_owner) {
            throw std::invalid_argument(
                "SparseMatrix requires a live RXMeshStatic owner.");
        }
        const DType parsed_dtype = parse_dtype(dtype);
        switch (parsed_dtype) {
            case DType::Float32:
                matrix = std::make_shared<rxmesh::SparseMatrix<float>>(
                    *mesh_owner, op);
                break;
            case DType::Float64:
                matrix = std::make_shared<rxmesh::SparseMatrix<double>>(
                    *mesh_owner, op);
                break;
            case DType::Int32:
                matrix = std::make_shared<rxmesh::SparseMatrix<int32_t>>(
                    *mesh_owner, op);
                break;
            default:
                throw std::invalid_argument(
                    "SparseMatrix supports float32, float64, and int32.");
        }
    }

    PySparseMatrix(SparseMatrixVariant matrix_in,
                   OwnedCsrVariant     owned_buffers_in,
                   rxmesh::Op          query_op,
                   rxmesh::locationT   location)
        : op(query_op),
          allocated(location),
          owned_buffers(std::move(owned_buffers_in)),
          matrix(std::move(matrix_in))
    {
    }

    virtual ~PySparseMatrix()
    {
        release();
    }

    std::shared_ptr<rxmesh::RXMeshStatic> mesh_owner;
    rxmesh::Op                            op;
    rxmesh::locationT                     allocated;
    OwnedCsrVariant                       owned_buffers = std::monostate{};
    SparseMatrixVariant                   matrix;
    bool                                  released = false;

    int rows() const
    {
        return std::visit([](const auto& mat) { return mat->rows(); }, matrix);
    }

    int cols() const
    {
        return std::visit([](const auto& mat) { return mat->cols(); }, matrix);
    }

    int nnz() const
    {
        return std::visit([](const auto& mat) { return mat->non_zeros(); },
                          matrix);
    }

    py::tuple shape() const
    {
        return py::make_tuple(rows(), cols());
    }

    std::string dtype() const
    {
        return std::visit(
            [](const auto& mat) {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                return std::string(sparse_dtype_name<T>());
            },
            matrix);
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

    int lower_nnz() const
    {
        ensure_host_readable();
        std::visit([](const auto& mat) { return mat->lower_non_zeros(); },
                   matrix);
    }

    void ensure_not_released() const
    {
        if (released) {
            throw std::runtime_error("SparseMatrix has been released.");
        }
    }

    void ensure_host_readable() const
    {
        ensure_not_released();
        if (!is_host_allocated()) {
            throw std::runtime_error(
                "SparseMatrix has no HOST allocation to read.");
        }
    }

    void ensure_device_readable()
    {
        ensure_not_released();
        if (!is_device_allocated()) {
            throw std::runtime_error(
                "SparseMatrix has no DEVICE allocation to read.");
        }
    }

    void move(int source, int target, py::object stream)
    {
        ensure_not_released();
        const auto src         = parse_location(source);
        const auto dst         = parse_location(target);
        const auto cuda_stream = parse_cuda_stream_arg(std::move(stream));
        std::visit([&](const auto& mat) { mat->move(src, dst, cuda_stream); },
                   matrix);
    }

    void reset(py::object value, int location, py::object stream)
    {
        ensure_not_released();
        const auto loc         = parse_location(location);
        const auto cuda_stream = parse_cuda_stream_arg(std::move(stream));
        std::visit(
            [&](const auto& mat) {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                mat->reset(value.cast<T>(), loc, cuda_stream);
            },
            matrix);
    }

    void copy_from(PySparseMatrix& other,
                   int             source,
                   int             target,
                   py::object      stream)
    {
        ensure_not_released();
        other.ensure_not_released();
        if (dtype() != other.dtype() || rows() != other.rows() ||
            cols() != other.cols() || nnz() != other.nnz()) {
            throw std::invalid_argument(
                "SparseMatrix.copy_from() requires matching dtype, shape, and "
                "nnz.");
        }

        const auto src         = parse_location(source);
        const auto dst         = parse_location(target);
        const auto cuda_stream = parse_cuda_stream_arg(std::move(stream));
        std::visit(
            [&](auto& dst_mat) {
                using DstMatT = std::decay_t<decltype(dst_mat)>;
                std::visit(
                    [&](auto& src_mat) {
                        using SrcMatT = std::decay_t<decltype(src_mat)>;
                        if constexpr (std::is_same_v<DstMatT, SrcMatT>) {
                            dst_mat->copy_from(*src_mat, src, dst, cuda_stream);
                        } else {
                            throw std::invalid_argument(
                                "SparseMatrix.copy_from() requires exactly "
                                "matching typed matrices.");
                        }
                    },
                    other.matrix);
            },
            matrix);
    }

    bool is_non_zero(int row, int col)
    {
        ensure_host_readable();
        return std::visit(
            [&](const auto& mat) { return mat->is_non_zero(row, col); },
            matrix);
    }

    py::object value(int row, int col)
    {
        ensure_host_readable();
        return std::visit(
            [&](const auto& mat) -> py::object {
                if (!mat->is_non_zero(row, col)) {
                    return py::cast(0);
                }
                return py::cast((*mat)(row, col));
            },
            matrix);
    }

    void set_value(int row, int col, py::object value)
    {
        ensure_host_readable();
        std::visit(
            [&](const auto& mat) {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                if (!mat->is_non_zero(row, col)) {
                    throw std::invalid_argument(
                        "SparseMatrix.set_value() can only update existing CSR "
                        "entries.");
                }
                (*mat)(row, col) = value.cast<T>();
            },
            matrix);
    }

    void sync_host_to_device(py::object stream)
    {
        using namespace rxmesh;
        move(static_cast<int>(HOST),
             static_cast<int>(DEVICE),
             std::move(stream));
    }

    void sync_device_to_host(py::object stream)
    {
        using namespace rxmesh;
        move(static_cast<int>(DEVICE),
             static_cast<int>(HOST),
             std::move(stream));
    }

    py::array row_ptr_to_numpy_impl(bool copy)
    {
        ensure_host_readable();
        return std::visit(
            [&](const auto& mat) -> py::array {
                using IndexT =
                    typename std::decay_t<decltype(mat)>::element_type::IndexT;
                return make_numpy_1d<IndexT>(
                    mat->row_ptr(rxmesh::HOST), rows() + 1, copy);
            },
            matrix);
    }

    py::array col_indices_to_numpy_impl(bool copy)
    {
        ensure_host_readable();
        return std::visit(
            [&](const auto& mat) -> py::array {
                using IndexT =
                    typename std::decay_t<decltype(mat)>::element_type::IndexT;
                return make_numpy_1d<IndexT>(
                    mat->col_idx(rxmesh::HOST), nnz(), copy);
            },
            matrix);
    }

    py::array values_to_numpy_impl(int source, bool copy, py::object stream)
    {
        const auto src = parse_location(source);
        if (src == rxmesh::DEVICE) {
            sync_device_to_host(stream);
            cuda_stream_synchronize_arg(std::move(stream));
        }
        ensure_host_readable();
        return std::visit(
            [&](const auto& mat) -> py::array {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                return make_numpy_1d<T>(
                    mat->val_ptr(rxmesh::HOST), nnz(), copy);
            },
            matrix);
    }

    py::tuple to_numpy(int location)
    {
        const auto loc = parse_location(location);
        if (loc != rxmesh::HOST) {
            throw std::invalid_argument(
                "SparseMatrix.to_numpy() only supports Location.HOST. Use "
                "SparseMatrix.to_torch(Location.DEVICE) for CUDA zero-copy "
                "views or SparseMatrix.to_numpy_copy() for copies.");
        }
        return py::make_tuple(
            row_ptr_to_numpy_impl(false),
            col_indices_to_numpy_impl(false),
            values_to_numpy_impl(
                static_cast<int>(rxmesh::HOST), false, py::none()));
    }

    py::tuple to_numpy_copy(int source, py::object stream)
    {
        const auto src = parse_location(source);
        if (src != rxmesh::HOST && src != rxmesh::DEVICE) {
            throw std::invalid_argument(
                "SparseMatrix.to_numpy_copy() source must be Location.HOST or "
                "Location.DEVICE.");
        }
        return py::make_tuple(
            row_ptr_to_numpy_impl(true),
            col_indices_to_numpy_impl(true),
            values_to_numpy_impl(source, true, std::move(stream)));
    }

    py::array values_to_numpy(int location)
    {
        const auto loc = parse_location(location);
        if (loc != rxmesh::HOST) {
            throw std::invalid_argument(
                "SparseMatrix.values_to_numpy() only supports Location.HOST. "
                "Use SparseMatrix.values_to_numpy_copy() for copies.");
        }
        return values_to_numpy_impl(
            static_cast<int>(rxmesh::HOST), false, py::none());
    }

    py::array values_to_numpy_copy(int source, py::object stream)
    {
        const auto src = parse_location(source);
        if (src != rxmesh::HOST && src != rxmesh::DEVICE) {
            throw std::invalid_argument(
                "SparseMatrix.values_to_numpy_copy() source must be "
                "Location.HOST or Location.DEVICE.");
        }
        return values_to_numpy_impl(source, true, std::move(stream));
    }

    void from_numpy_values_copy(py::array values, int target, py::object stream)
    {
        ensure_host_readable();
        std::visit(
            [&](const auto& mat) {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                py::array_t<T, py::array::c_style | py::array::forcecast> typed(
                    values);
                const py::buffer_info info = typed.request();
                if (info.ndim != 1 ||
                    info.shape[0] != static_cast<py::ssize_t>(nnz())) {
                    throw std::invalid_argument(
                        "SparseMatrix.from_numpy_values_copy() expects a 1D "
                        "array with length nnz.");
                }
                auto view = typed.template unchecked<1>();
                T*   dst  = mat->val_ptr(rxmesh::HOST);
                for (int i = 0; i < nnz(); ++i) {
                    dst[i] = view(i);
                }
            },
            matrix);

        if ((parse_location(target) & rxmesh::DEVICE) == rxmesh::DEVICE) {
            sync_host_to_device(std::move(stream));
        }
    }

    std::shared_ptr<PyDenseMatrix> multiply_dense(PyDenseMatrix& rhs,
                                                  bool           transpose_a,
                                                  bool           transpose_b,
                                                  py::object     alpha,
                                                  py::object     beta,
                                                  py::object     stream)
    {
        using namespace rxmesh;
        ensure_not_released();
        ensure_device_readable();
        const auto cuda_stream = parse_cuda_stream_arg(std::move(stream));

        const int a_rows = transpose_a ? cols() : rows();
        const int a_cols = transpose_a ? rows() : cols();
        const int b_rows = transpose_b ? rhs.cols() : rhs.rows();
        const int b_cols = transpose_b ? rhs.rows() : rhs.cols();
        if (a_cols != b_rows) {
            throw std::invalid_argument(
                "SparseMatrix.multiply() shape mismatch between sparse matrix "
                "and dense operand.");
        }

        auto output = std::make_shared<PyDenseMatrix>(
            rhs.dtype(),
            a_rows,
            b_cols,
            static_cast<int>(rxmesh::LOCATION_ALL),
            "col_major");

        output->reset(py::float_(0.0), rxmesh::DEVICE, cuda_stream);

        std::visit(
            [&](auto& sparse_mat) {
                using SparseMatT = std::decay_t<decltype(sparse_mat)>;
                using T          = typename SparseMatT::element_type::Type;

                std::visit(
                    [&](auto& rhs_mat) {
                        using RhsMatT = std::decay_t<decltype(rhs_mat)>;
                        std::visit(
                            [&](auto& out_mat) {
                                using OutMatT = std::decay_t<decltype(out_mat)>;
                                if constexpr (
                                    std::is_same_v<
                                        T,
                                        typename RhsMatT::element_type::Type> &&
                                    std::is_same_v<
                                        T,
                                        typename OutMatT::element_type::Type>) {
                                    sparse_mat->multiply(*rhs_mat,
                                                         *out_mat,
                                                         transpose_a,
                                                         transpose_b,
                                                         alpha.cast<T>(),
                                                         beta.cast<T>(),
                                                         cuda_stream);
                                } else {
                                    throw std::invalid_argument(
                                        "SparseMatrix.multiply() requires "
                                        "exactly matching typed matrices.");
                                }
                            },
                            output->matrix);
                    },
                    rhs.matrix);
            },
            matrix);

        output->move(rxmesh::DEVICE, rxmesh::HOST, cuda_stream);
        return output;
    }

    std::shared_ptr<PyDenseMatrix> multiply_vector(PyDenseMatrix& rhs,
                                                   py::object     stream)
    {
        using namespace rxmesh;
        ensure_not_released();
        ensure_device_readable();
        const auto cuda_stream = parse_cuda_stream_arg(std::move(stream));

        if (rhs.cols() != 1) {
            throw std::invalid_argument(
                "SparseMatrix.multiply_vector() requires a DenseMatrix with "
                "one column.");
        }

        auto output = std::make_shared<PyDenseMatrix>(
            rhs.dtype(),
            rows(),
            1,
            static_cast<int>(rxmesh::LOCATION_ALL),
            "col_major");

        std::visit(
            [&](auto& sparse_mat) {
                using SparseMatT = std::decay_t<decltype(sparse_mat)>;
                using T          = typename SparseMatT::element_type::Type;

                std::visit(
                    [&](auto& rhs_mat) {
                        using RhsMatT = std::decay_t<decltype(rhs_mat)>;
                        std::visit(
                            [&](auto& out_mat) {
                                using OutMatT = std::decay_t<decltype(out_mat)>;
                                if constexpr (
                                    std::is_same_v<
                                        T,
                                        typename RhsMatT::element_type::Type> &&
                                    std::is_same_v<
                                        T,
                                        typename OutMatT::element_type::Type>) {
                                    sparse_mat->multiply_cw(
                                        *rhs_mat, *out_mat, cuda_stream);
                                } else {
                                    throw std::invalid_argument(
                                        "SparseMatrix.multiply_vector() "
                                        "requires exactly matching typed "
                                        "matrices.");
                                }
                            },
                            output->matrix);
                    },
                    rhs.matrix);
            },
            matrix);

        output->move(rxmesh::DEVICE, rxmesh::HOST, cuda_stream);
        return output;
    }

    std::shared_ptr<PySparseMatrix> transpose(py::object stream)
    {
        using namespace rxmesh;
        ensure_not_released();
        ensure_device_readable();
        const auto cuda_stream = parse_cuda_stream_arg(std::move(stream));
        if (op == rxmesh::Op::INVALID) {
            throw std::invalid_argument(
                "SparseMatrix.transpose() currently supports matrices built "
                "from RXMesh query ops.");
        }
        return std::visit(
            [&](const auto& mat) -> std::shared_ptr<PySparseMatrix> {
                using MatT      = std::decay_t<decltype(mat)>;
                using T         = typename MatT::element_type::Type;
                auto transposed = std::make_shared<rxmesh::SparseMatrix<T>>(
                    std::move(mat->transpose()));
                make_stream_wait_for_legacy_default(cuda_stream);
                using IndexT = typename rxmesh::SparseMatrix<T>::IndexT;
                CUDA_ERROR(cudaMemcpyAsync(
                    transposed->row_ptr(rxmesh::HOST),
                    transposed->row_ptr(rxmesh::DEVICE),
                    static_cast<size_t>(transposed->rows() + 1) *
                        sizeof(IndexT),
                    cudaMemcpyDeviceToHost,
                    cuda_stream));
                CUDA_ERROR(cudaMemcpyAsync(
                    transposed->col_idx(rxmesh::HOST),
                    transposed->col_idx(rxmesh::DEVICE),
                    static_cast<size_t>(transposed->non_zeros()) *
                        sizeof(IndexT),
                    cudaMemcpyDeviceToHost,
                    cuda_stream));
                CUDA_ERROR(cudaMemcpyAsync(
                    transposed->val_ptr(rxmesh::HOST),
                    transposed->val_ptr(rxmesh::DEVICE),
                    static_cast<size_t>(transposed->non_zeros()) * sizeof(T),
                    cudaMemcpyDeviceToHost,
                    cuda_stream));
                SparseMatrixVariant wrapped = transposed;
                return std::make_shared<PySparseMatrix>(
                    std::move(wrapped),
                    OwnedCsrVariant{std::monostate{}},
                    transposed->get_op(),
                    rxmesh::LOCATION_ALL);
            },
            matrix);
    }

    void to_mtx(const std::string& file_name)
    {
        ensure_host_readable();
        std::visit([&](const auto& mat) { mat->to_mtx(file_name); }, matrix);
    }

    void to_file(const std::string& file_name)
    {
        ensure_host_readable();
        std::visit([&](const auto& mat) { mat->to_file(file_name); }, matrix);
    }

    void release()
    {
        if (released) {
            return;
        }
        std::visit([](const auto& mat) { mat->release(); }, matrix);
        released  = true;
        allocated = rxmesh::LOCATION_NONE;
    }

   private:
    template <typename T>
    py::array make_numpy_1d(T* data, int64_t size, bool copy)
    {
        if (copy) {
            py::array_t<T> out(static_cast<py::ssize_t>(size));
            auto           view = out.template mutable_unchecked<1>();
            for (int64_t i = 0; i < size; ++i) {
                view(static_cast<py::ssize_t>(i)) = data[i];
            }
            return out;
        }
        return py::array_t<T>({static_cast<py::ssize_t>(size)},
                              {static_cast<py::ssize_t>(sizeof(T))},
                              data,
                              py::cast(shared_from_this()));
    }
};

}  // namespace pyrxmesh_py
