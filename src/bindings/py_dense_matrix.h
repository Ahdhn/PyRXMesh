#pragma once

#include "bindings/common.h"
#include "bindings/dispatch.h"

namespace pyrxmesh_py {

// Forward decl of the typed implementation so factory helpers can construct
// it from per-T context (DLPack copy, sparse multiply output, attribute
// to_matrix_copy, dense view creation, etc.)
template <typename T>
struct PyDenseMatrixT;


// -----------------------------------------------------------------------------
// PyDenseMatrix — non-templated polymorphic base.
//
// Mirrors the PySparseMatrix / PyAttributeBase pattern: dtype dispatch happens
// at construction time (factory picks PyDenseMatrixT<T>), and all subsequent
// operations go through virtual methods on the base. Binary operations
// resolve their other operand via dynamic_cast<PyDenseMatrixT<T>*>.
// -----------------------------------------------------------------------------
struct PyDenseMatrix : std::enable_shared_from_this<PyDenseMatrix>
{
    rxmesh::locationT              allocated = rxmesh::LOCATION_NONE;
    std::shared_ptr<PyDenseMatrix> base_owner;

    virtual ~PyDenseMatrix() = default;

    // Metadata
    virtual int         rows() const  = 0;
    virtual int         cols() const  = 0;
    virtual int         bytes() const = 0;
    virtual std::string dtype() const = 0;

    // Shared (non-virtual) helpers
    py::tuple shape() const
    {
        return py::make_tuple(rows(), cols());
    }
    std::string order() const
    {
        return "col_major";
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

    // Allocation lifecycle
    virtual void move(rxmesh::locationT source,
                      rxmesh::locationT target,
                      cudaStream_t      stream) = 0;
    virtual void release(int location)     = 0;

    // Value ops
    virtual void reset(py::object value, int location, cudaStream_t stream) = 0;
    virtual void fill_random(double low, double high)                       = 0;
    virtual py::object value(py::object row_or_handle, int col)             = 0;
    virtual void       set_value(py::object row_or_handle,
                                 int        col,
                                 py::object value)                          = 0;

    // NumPy
    virtual py::array to_numpy(int location)               = 0;
    virtual py::array to_numpy_copy(int source)            = 0;
    virtual void      from_numpy_copy(py::array    values,
                                      int          target,
                                      cudaStream_t stream) = 0;
    virtual void      copy_from(PyDenseMatrix& other,
                                int            source,
                                int            target,
                                cudaStream_t   stream)       = 0;

    // BLAS-like
    virtual py::object norm2(cudaStream_t stream)                       = 0;
    virtual py::object abs_sum(cudaStream_t stream)                     = 0;
    virtual py::object abs_max(cudaStream_t stream)                     = 0;
    virtual py::object abs_min(cudaStream_t stream)                     = 0;
    virtual py::object dot(PyDenseMatrix& other, cudaStream_t stream)   = 0;
    virtual void       axpy(PyDenseMatrix& x,
                            py::object     alpha,
                            cudaStream_t   stream)                        = 0;
    virtual void       multiply(py::object scalar, cudaStream_t stream) = 0;
    virtual void       swap(PyDenseMatrix& other, cudaStream_t stream)  = 0;

    // Shape / view operations
    virtual void                           reshape(int rows, int cols)   = 0;
    virtual std::shared_ptr<PyDenseMatrix> col(int column)               = 0;
    virtual std::shared_ptr<PyDenseMatrix> segment(int start, int count) = 0;

    // File I/O
    virtual void to_mtx(const std::string& file_name) = 0;
};


// -----------------------------------------------------------------------------
// PyDenseMatrixT<T> — typed implementation. T must be float, double, or
// int32_t.
// -----------------------------------------------------------------------------
template <typename T>
struct PyDenseMatrixT final : PyDenseMatrix
{
    using MatT = rxmesh::DenseMatrix<T, Eigen::ColMajor>;

    std::shared_ptr<MatT> matrix;

    PyDenseMatrixT() = default;

    PyDenseMatrixT(std::shared_ptr<MatT> in_matrix, rxmesh::locationT location)
    {
        matrix    = std::move(in_matrix);
        allocated = location;
    }

    ~PyDenseMatrixT() override
    {
        if (matrix) {
            matrix->release(rxmesh::LOCATION_ALL);
        }
    }

    // Metadata
    int rows() const override
    {
        return matrix->rows();
    }
    int cols() const override
    {
        return matrix->cols();
    }
    int bytes() const override
    {
        return matrix->bytes();
    }
    std::string dtype() const override
    {
        return std::string(dense_dtype_name<T>());
    }

    // Allocation
    void move(rxmesh::locationT source,
              rxmesh::locationT target,
              cudaStream_t      stream) override
    {
        matrix->move(source, target, stream);
        allocated = static_cast<rxmesh::locationT>(static_cast<int>(allocated) |
                                                   static_cast<int>(target));
    }

    void release(int location) override
    {
        const auto loc = parse_location(location);
        matrix->release(loc);
        allocated = static_cast<rxmesh::locationT>(static_cast<int>(allocated) &
                                                   (~static_cast<int>(loc)));
    }

    // Value ops
    void reset(py::object value, int location, cudaStream_t stream) override
    {
        matrix->reset(value.cast<T>(), parse_location(location), stream);
    }

    void fill_random(double low, double high) override
    {
        matrix->fill_random(low, high);
    }

    py::object value(py::object row_or_handle, int col) override
    {
        return py::cast(read_value(row_or_handle, col));
    }

    void set_value(py::object row_or_handle,
                   int        col,
                   py::object value_obj) override
    {
        write_value(row_or_handle, col, value_obj.cast<T>());
    }

    // NumPy
    py::array to_numpy(int location) override
    {
        const auto loc = parse_location(location);
        if (loc != rxmesh::HOST) {
            throw std::invalid_argument(
                "DenseMatrix.to_numpy() only supports Location.HOST.");
        }
        if (!is_host_allocated()) {
            throw std::runtime_error(
                "DenseMatrix.to_numpy() requires an existing HOST allocation. "
                "Move or create the matrix on HOST, or use "
                "DenseMatrix.to_numpy_copy().");
        }
        const int r     = matrix->rows();
        const int c     = matrix->cols();
        auto      owner = shared_from_this();
        return py::array_t<T>({r, c},
                              {static_cast<py::ssize_t>(sizeof(T)),
                               static_cast<py::ssize_t>(sizeof(T) * r)},
                              matrix->data(rxmesh::HOST),
                              py::cast(owner));
    }

    py::array to_numpy_copy(int source) override
    {
        const auto src = parse_location(source);
        if (src != rxmesh::HOST) {
            throw std::invalid_argument(
                "DenseMatrix.to_numpy_copy() only supports Location.HOST.");
        }
        const int      r = matrix->rows();
        const int      c = matrix->cols();
        py::array_t<T> out({r, c});
        auto           view = out.template mutable_unchecked<2>();
        for (int j = 0; j < c; ++j) {
            for (int i = 0; i < r; ++i) {
                view(i, j) = (*matrix)(i, j);
            }
        }
        return out;
    }

    void from_numpy_copy(py::array    values,
                         int          target,
                         cudaStream_t stream) override
    {
        const auto dst = parse_location(target);

        py::array_t<T, py::array::c_style | py::array::forcecast> typed(values);
        const py::buffer_info info = typed.request();
        if (info.ndim != 2 || info.shape[0] != matrix->rows() ||
            info.shape[1] != matrix->cols()) {
            throw std::invalid_argument(
                "DenseMatrix.from_numpy_copy() shape must match matrix.shape.");
        }
        auto view = typed.template unchecked<2>();
        for (int j = 0; j < matrix->cols(); ++j) {
            for (int i = 0; i < matrix->rows(); ++i) {
                (*matrix)(i, j) = view(i, j);
            }
        }

        if ((dst & rxmesh::DEVICE) == rxmesh::DEVICE) {
            move(rxmesh::HOST, rxmesh::DEVICE, stream);
        }
    }

    void copy_from(PyDenseMatrix& other,
                   int            source,
                   int            target,
                   cudaStream_t   stream) override
    {
        auto* typed = dynamic_cast<PyDenseMatrixT<T>*>(&other);
        if (!typed) {
            throw std::invalid_argument(
                "DenseMatrix.copy_from() requires exactly matching typed "
                "matrices.");
        }
        const auto src = parse_location(source);
        const auto dst = parse_location(target);
        matrix->copy_from(*typed->matrix, src, dst, stream);
        allocated = static_cast<rxmesh::locationT>(static_cast<int>(allocated) |
                                                   static_cast<int>(dst));
    }

    // BLAS-like (float/double only)
    py::object norm2(cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.norm2() supports float32 and float64.");
        } else {
            return py::cast(matrix->norm2(stream));
        }
    }

    py::object abs_sum(cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.abs_sum() supports float32 and float64.");
        } else {
            return py::cast(matrix->abs_sum(stream));
        }
    }

    py::object abs_max(cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.abs_max() supports float32 and float64.");
        } else {
            return py::cast(matrix->abs_max(stream));
        }
    }

    py::object abs_min(cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.abs_min() supports float32 and float64.");
        } else {
            return py::cast(matrix->abs_min(stream));
        }
    }

    py::object dot(PyDenseMatrix& other, cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.dot() supports float32 and float64.");
        } else {
            auto* typed = dynamic_cast<PyDenseMatrixT<T>*>(&other);
            if (!typed) {
                throw std::invalid_argument(
                    "DenseMatrix.dot() requires exactly matching typed "
                    "matrices.");
            }
            if (rows() != typed->rows() || cols() != typed->cols()) {
                throw std::invalid_argument(
                    "DenseMatrix.dot() requires matching dtype and shape.");
            }
            return py::cast(matrix->dot(*typed->matrix, false, stream));
        }
    }

    void axpy(PyDenseMatrix& x, py::object alpha, cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.axpy() supports float32 and float64.");
        } else {
            auto* typed = dynamic_cast<PyDenseMatrixT<T>*>(&x);
            if (!typed) {
                throw std::invalid_argument(
                    "DenseMatrix.axpy() requires matching dtype.");
            }
            matrix->axpy(*typed->matrix, alpha.cast<T>(), stream);
        }
    }

    void multiply(py::object scalar, cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.multiply() supports float32 and float64.");
        } else {
            matrix->multiply(scalar.cast<T>(), stream);
        }
    }

    void swap(PyDenseMatrix& other, cudaStream_t stream) override
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            throw std::invalid_argument(
                "DenseMatrix.swap() supports float32 and float64.");
        } else {
            auto* typed = dynamic_cast<PyDenseMatrixT<T>*>(&other);
            if (!typed) {
                throw std::invalid_argument(
                    "DenseMatrix.swap() requires matching dtype.");
            }
            matrix->swap(*typed->matrix, stream);
        }
    }

    // Shape / view ops
    void reshape(int new_rows, int new_cols) override
    {
        if (new_rows <= 0 || new_cols <= 0 ||
            new_rows * new_cols != rows() * cols()) {
            throw std::invalid_argument(
                "DenseMatrix.reshape() must preserve element count.");
        }
        matrix->reshape(new_rows, new_cols);
    }

    std::shared_ptr<PyDenseMatrix> col(int column) override
    {
        if (column < 0 || column >= cols()) {
            throw std::out_of_range(
                "DenseMatrix.col() column is out of range.");
        }
        auto view = std::make_shared<MatT>(matrix->col(column));
        auto ret =
            std::make_shared<PyDenseMatrixT<T>>(std::move(view), allocated);
        ret->base_owner = shared_from_this();
        return ret;
    }

    std::shared_ptr<PyDenseMatrix> segment(int start, int count) override
    {
        if (start < 0 || count < 0 || start + count > rows() * cols()) {
            throw std::out_of_range(
                "DenseMatrix.segment() range is out of bounds.");
        }
        auto view = std::make_shared<MatT>(matrix->segment(start, count));
        auto ret =
            std::make_shared<PyDenseMatrixT<T>>(std::move(view), allocated);
        ret->base_owner = shared_from_this();
        return ret;
    }

    // File I/O
    void to_mtx(const std::string& file_name) override
    {
        matrix->to_mtx(file_name);
    }

   private:
    int validate_row(int row) const
    {
        if (row < 0 || row >= rows()) {
            throw std::out_of_range("DenseMatrix row index is out of range.");
        }
        return row;
    }

    T read_value(py::object row_or_handle, int col)
    {
        if (py::isinstance<py::int_>(row_or_handle)) {
            return (*matrix)(validate_row(row_or_handle.cast<int>()), col);
        }
        if (py::isinstance<rxmesh::VertexHandle>(row_or_handle)) {
            return (*matrix)(row_or_handle.cast<rxmesh::VertexHandle>(), col);
        }
        if (py::isinstance<rxmesh::EdgeHandle>(row_or_handle)) {
            return (*matrix)(row_or_handle.cast<rxmesh::EdgeHandle>(), col);
        }
        if (py::isinstance<rxmesh::FaceHandle>(row_or_handle)) {
            return (*matrix)(row_or_handle.cast<rxmesh::FaceHandle>(), col);
        }
        throw py::type_error(
            "DenseMatrix row must be an int or RXMesh handle.");
    }

    void write_value(py::object row_or_handle, int col, T value)
    {
        if (py::isinstance<py::int_>(row_or_handle)) {
            (*matrix)(validate_row(row_or_handle.cast<int>()), col) = value;
            return;
        }
        if (py::isinstance<rxmesh::VertexHandle>(row_or_handle)) {
            (*matrix)(row_or_handle.cast<rxmesh::VertexHandle>(), col) = value;
            return;
        }
        if (py::isinstance<rxmesh::EdgeHandle>(row_or_handle)) {
            (*matrix)(row_or_handle.cast<rxmesh::EdgeHandle>(), col) = value;
            return;
        }
        if (py::isinstance<rxmesh::FaceHandle>(row_or_handle)) {
            (*matrix)(row_or_handle.cast<rxmesh::FaceHandle>(), col) = value;
            return;
        }
        throw py::type_error(
            "DenseMatrix row must be an int or RXMesh handle.");
    }
};


inline void validate_dense_matrix_shape(int rows, int cols)
{
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "DenseMatrix rows and cols must be positive.");
    }
}

inline void validate_dense_matrix_order(const std::string& order)
{
    if (order != "col_major" && order != "column_major" && order != "F") {
        throw std::invalid_argument(
            "DenseMatrix currently supports only col_major order.");
    }
}

inline std::shared_ptr<PyDenseMatrix> make_dense_matrix(
    int                rows,
    int                cols,
    const std::string& dtype,
    int                location,
    const std::string& order)
{
    validate_dense_matrix_shape(rows, cols);
    validate_dense_matrix_order(order);
    const auto loc = parse_location(location);
    return dispatch_numeric_dtype_str(
        dtype, [&](auto tag) -> std::shared_ptr<PyDenseMatrix> {
            using T = typename decltype(tag)::type;
            using MatT =
                typename PyDenseMatrixT<T>::MatT;  // DenseMatrix<T, ColMajor>
            return std::make_shared<PyDenseMatrixT<T>>(
                std::make_shared<MatT>(rows, cols, loc), loc);
        });
}

inline std::shared_ptr<PyDenseMatrix> make_dense_matrix_for_mesh(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    int                                   rows,
    int                                   cols,
    const std::string&                    dtype,
    int                                   location,
    const std::string&                    order)
{
    if (!mesh) {
        throw std::invalid_argument(
            "DenseMatrix mesh-aware constructor requires a mesh.");
    }
    validate_dense_matrix_shape(rows, cols);
    validate_dense_matrix_order(order);
    const auto loc = parse_location(location);
    return dispatch_numeric_dtype_str(
        dtype, [&](auto tag) -> std::shared_ptr<PyDenseMatrix> {
            using T    = typename decltype(tag)::type;
            using MatT = typename PyDenseMatrixT<T>::MatT;
            return std::make_shared<PyDenseMatrixT<T>>(
                std::make_shared<MatT>(*mesh, rows, cols, loc), loc);
        });
}

/**
 * Cast a PyDenseMatrix to its typed implementation, throwing error
 * if the dtype does not match. Mirrors with_float_double_sparse_matrix in
 * py_sparse_matrix.h. Used by sparse-dense multiply, attribute<->matrix
 * bridges, DLPack import, and solver typed access paths.
 */
template <typename T>
inline PyDenseMatrixT<T>& as_typed_dense(PyDenseMatrix& mat,
                                         const char*    api_name)
{
    auto* typed = dynamic_cast<PyDenseMatrixT<T>*>(&mat);
    if (!typed) {
        throw std::invalid_argument(
            std::string(api_name) +
            " requires a DenseMatrix with matching dtype.");
    }
    return *typed;
}

template <typename T>
inline const PyDenseMatrixT<T>& as_typed_dense(const PyDenseMatrix& mat,
                                               const char*          api_name)
{
    const auto* typed = dynamic_cast<const PyDenseMatrixT<T>*>(&mat);
    if (!typed) {
        throw std::invalid_argument(
            std::string(api_name) +
            " requires a DenseMatrix with matching dtype.");
    }
    return *typed;
}

/**
 * Dispatch to the typed PyDenseMatrixT<T> implementation by `dynamic_cast`,
 * invoking `fn(typed)` with the matching reference. Used by code that needs
 * to be polymorphic over the supported dtypes (float/double/int32) without
 * knowing T at the call site, e.g. DLPack export, sparse-dense multiply.
 */
template <typename Fn>
auto with_typed_dense_matrix(PyDenseMatrix& self, Fn&& fn)
    -> decltype(std::forward<Fn>(fn)(std::declval<PyDenseMatrixT<float>&>()))
{
    if (auto* p = dynamic_cast<PyDenseMatrixT<float>*>(&self)) {
        return std::forward<Fn>(fn)(*p);
    }
    if (auto* p = dynamic_cast<PyDenseMatrixT<double>*>(&self)) {
        return std::forward<Fn>(fn)(*p);
    }
    if (auto* p = dynamic_cast<PyDenseMatrixT<int32_t>*>(&self)) {
        return std::forward<Fn>(fn)(*p);
    }
    throw std::runtime_error("DenseMatrix has an unknown dtype.");
}

}  // namespace pyrxmesh_py
