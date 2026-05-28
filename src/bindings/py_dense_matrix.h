#pragma once

#include "bindings/common.h"

namespace pyrxmesh_py {

using DenseMatrixVariant = std::variant<
    std::shared_ptr<rxmesh::DenseMatrix<float, Eigen::ColMajor>>,
    std::shared_ptr<rxmesh::DenseMatrix<double, Eigen::ColMajor>>,
    std::shared_ptr<rxmesh::DenseMatrix<int32_t, Eigen::ColMajor>>>;

struct PyDenseMatrix : std::enable_shared_from_this<PyDenseMatrix>
{
    PyDenseMatrix(std::string dtype,
                  int         rows,
                  int         cols,
                  int         location,
                  std::string order)
        : allocated(parse_location(location))
    {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument(
                "DenseMatrix rows and cols must be positive.");
        }
        if (order != "col_major" && order != "column_major" && order != "F") {
            throw std::invalid_argument(
                "DenseMatrix currently supports only col_major order.");
        }

        const DType parsed_dtype = parse_dtype(dtype);
        const auto  loc          = parse_location(location);
        switch (parsed_dtype) {
            case DType::Float32:
                matrix = std::make_shared<
                    rxmesh::DenseMatrix<float, Eigen::ColMajor>>(
                    rows, cols, loc);
                break;
            case DType::Float64:
                matrix = std::make_shared<
                    rxmesh::DenseMatrix<double, Eigen::ColMajor>>(
                    rows, cols, loc);
                break;
            case DType::Int32:
                matrix = std::make_shared<
                    rxmesh::DenseMatrix<int32_t, Eigen::ColMajor>>(
                    rows, cols, loc);
                break;
            default:
                throw std::invalid_argument(
                    "DenseMatrix supports float32, float64, and int32.");
        }
    }

    explicit PyDenseMatrix(DenseMatrixVariant mat, rxmesh::locationT location)
        : allocated(location), matrix(std::move(mat))
    {
    }

    ~PyDenseMatrix()
    {
        std::visit([](const auto& mat) { mat->release(rxmesh::LOCATION_ALL); },
                   matrix);
    }

    rxmesh::locationT              allocated;
    DenseMatrixVariant             matrix;
    std::shared_ptr<PyDenseMatrix> base_owner;

    int rows() const
    {
        return std::visit([](const auto& mat) { return mat->rows(); }, matrix);
    }

    int cols() const
    {
        return std::visit([](const auto& mat) { return mat->cols(); }, matrix);
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
                return std::string(dense_dtype_name<T>());
            },
            matrix);
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

    int bytes() const
    {
        return std::visit([](const auto& mat) { return mat->bytes(); }, matrix);
    }

    void ensure_allocated(rxmesh::locationT location)
    {
        if (((location & rxmesh::HOST) == rxmesh::HOST) &&
            !is_host_allocated()) {
            if (!is_device_allocated()) {
                throw std::runtime_error(
                    "DenseMatrix has no allocation to copy from while ensuring "
                    "HOST.");
            }
            move(rxmesh::DEVICE, rxmesh::HOST);
        }
        if (((location & rxmesh::DEVICE) == rxmesh::DEVICE) &&
            !is_device_allocated()) {
            if (!is_host_allocated()) {
                throw std::runtime_error(
                    "DenseMatrix has no allocation to copy from while ensuring "
                    "DEVICE.");
            }
            move(rxmesh::HOST, rxmesh::DEVICE);
        }
    }

    void move(int source, int target)
    {
        move(parse_location(source), parse_location(target));
    }

    void move(rxmesh::locationT source, rxmesh::locationT target)
    {
        std::visit([&](const auto& mat) { mat->move(source, target); }, matrix);
        synchronize_device_transfer(source, target);
        allocated = static_cast<rxmesh::locationT>(static_cast<int>(allocated) |
                                                   static_cast<int>(target));
    }

    void release(int location)
    {
        const auto loc = parse_location(location);
        std::visit([&](const auto& mat) { mat->release(loc); }, matrix);
        allocated = static_cast<rxmesh::locationT>(static_cast<int>(allocated) &
                                                   (~static_cast<int>(loc)));
    }

    void reset(py::object value, int location)
    {
        const auto loc = parse_location(location);
        ensure_allocated(loc);
        std::visit(
            [&](const auto& mat) {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                mat->reset(value.cast<T>(), loc);
            },
            matrix);
    }

    void fill_random(double low, double high)
    {
        ensure_allocated(rxmesh::HOST);
        ensure_allocated(rxmesh::DEVICE);
        std::visit([&](const auto& mat) { mat->fill_random(low, high); },
                   matrix);
    }

    py::array to_numpy(int source, bool copy)
    {
        const auto src = parse_location(source);
        if ((src & rxmesh::HOST) != rxmesh::HOST) {
            throw std::invalid_argument(
                "DenseMatrix.to_numpy() only supports Location.HOST.");
        }
        ensure_allocated(rxmesh::HOST);

        return std::visit(
            [&](const auto& mat) -> py::array {
                using MatT  = std::decay_t<decltype(mat)>;
                using T     = typename MatT::element_type::Type;
                const int r = mat->rows();
                const int c = mat->cols();
                if (copy) {
                    py::array_t<T> out({r, c});
                    auto           view = out.template mutable_unchecked<2>();
                    for (int j = 0; j < c; ++j) {
                        for (int i = 0; i < r; ++i) {
                            view(i, j) = (*mat)(i, j);
                        }
                    }
                    return out;
                }

                auto owner = shared_from_this();
                return py::array_t<T>({r, c},
                                      {static_cast<py::ssize_t>(sizeof(T)),
                                       static_cast<py::ssize_t>(sizeof(T) * r)},
                                      mat->data(rxmesh::HOST),
                                      py::cast(owner));
            },
            matrix);
    }

    void from_numpy(py::array values, int target)
    {
        const auto dst = parse_location(target);
        ensure_allocated(rxmesh::HOST);

        std::visit(
            [&](const auto& mat) {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                py::array_t<T, py::array::c_style | py::array::forcecast> typed(
                    values);
                const py::buffer_info info = typed.request();
                if (info.ndim != 2 || info.shape[0] != mat->rows() ||
                    info.shape[1] != mat->cols()) {
                    throw std::invalid_argument(
                        "DenseMatrix.from_numpy() shape must match "
                        "matrix.shape.");
                }
                auto view = typed.template unchecked<2>();
                for (int j = 0; j < mat->cols(); ++j) {
                    for (int i = 0; i < mat->rows(); ++i) {
                        (*mat)(i, j) = view(i, j);
                    }
                }
            },
            matrix);

        if ((dst & rxmesh::DEVICE) == rxmesh::DEVICE) {
            ensure_allocated(rxmesh::DEVICE);
            move(rxmesh::HOST, rxmesh::DEVICE);
        }
    }

    void copy_from(PyDenseMatrix& other, int source, int target)
    {
        if (dtype() != other.dtype() || rows() != other.rows() ||
            cols() != other.cols()) {
            throw std::invalid_argument(
                "DenseMatrix.copy_from() requires matching dtype and shape.");
        }

        const auto src = parse_location(source);
        const auto dst = parse_location(target);
        other.ensure_allocated(src);
        ensure_allocated(dst);

        std::visit(
            [&](auto& dst_mat) {
                using DstMatT = std::decay_t<decltype(dst_mat)>;
                std::visit(
                    [&](auto& src_mat) {
                        using SrcMatT = std::decay_t<decltype(src_mat)>;
                        if constexpr (std::is_same_v<DstMatT, SrcMatT>) {
                            dst_mat->copy_from(*src_mat, src, dst);
                            synchronize_device_transfer(src, dst);
                            allocated = static_cast<rxmesh::locationT>(
                                static_cast<int>(allocated) |
                                static_cast<int>(dst));
                        } else {
                            throw std::invalid_argument(
                                "DenseMatrix.copy_from() requires exactly "
                                "matching typed matrices.");
                        }
                    },
                    other.matrix);
            },
            matrix);
    }

    py::object norm2()
    {
        ensure_allocated(rxmesh::DEVICE);
        return std::visit(
            [&](const auto& mat) -> py::object {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                if constexpr (std::is_same_v<T, int32_t>) {
                    throw std::invalid_argument(
                        "DenseMatrix.norm2() supports float32 and float64.");
                } else {
                    return py::cast(mat->norm2());
                }
            },
            matrix);
    }

    py::object abs_sum()
    {
        ensure_allocated(rxmesh::DEVICE);
        return std::visit(
            [&](const auto& mat) -> py::object {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                if constexpr (std::is_same_v<T, int32_t>) {
                    throw std::invalid_argument(
                        "DenseMatrix.abs_sum() supports float32 and float64.");
                } else {
                    return py::cast(mat->abs_sum());
                }
            },
            matrix);
    }

    py::object abs_max()
    {
        ensure_allocated(rxmesh::DEVICE);
        return std::visit(
            [&](const auto& mat) -> py::object {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                if constexpr (std::is_same_v<T, int32_t>) {
                    throw std::invalid_argument(
                        "DenseMatrix.abs_max() supports float32 and float64.");
                } else {
                    return py::cast(mat->abs_max());
                }
            },
            matrix);
    }

    py::object abs_min()
    {
        ensure_allocated(rxmesh::DEVICE);
        move(rxmesh::DEVICE, rxmesh::HOST);
        return std::visit(
            [&](const auto& mat) -> py::object {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                if constexpr (std::is_same_v<T, int32_t>) {
                    throw std::invalid_argument(
                        "DenseMatrix.abs_min() supports float32 and float64.");
                } else {
                    T result = std::numeric_limits<T>::max();
                    for (int j = 0; j < mat->cols(); ++j) {
                        for (int i = 0; i < mat->rows(); ++i) {
                            result =
                                std::min<T>(result, std::abs((*mat)(i, j)));
                        }
                    }
                    return py::cast(result);
                }
            },
            matrix);
    }

    py::object dot(PyDenseMatrix& other)
    {
        if (dtype() != other.dtype() || rows() != other.rows() ||
            cols() != other.cols()) {
            throw std::invalid_argument(
                "DenseMatrix.dot() requires matching dtype and shape.");
        }
        ensure_allocated(rxmesh::DEVICE);
        other.ensure_allocated(rxmesh::DEVICE);

        return std::visit(
            [&](const auto& lhs) -> py::object {
                using LhsMatT = std::decay_t<decltype(lhs)>;
                return std::visit(
                    [&](const auto& rhs) -> py::object {
                        using RhsMatT = std::decay_t<decltype(rhs)>;
                        if constexpr (std::is_same_v<LhsMatT, RhsMatT>) {
                            using T = typename LhsMatT::element_type::Type;
                            if constexpr (std::is_same_v<T, int32_t>) {
                                throw std::invalid_argument(
                                    "DenseMatrix.dot() supports float32 and "
                                    "float64.");
                            } else {
                                return py::cast(lhs->dot(*rhs));
                            }
                        } else {
                            throw std::invalid_argument(
                                "DenseMatrix.dot() requires exactly matching "
                                "typed matrices.");
                        }
                    },
                    other.matrix);
            },
            matrix);
    }

    void axpy(PyDenseMatrix& x, py::object alpha)
    {
        if (dtype() != x.dtype() || rows() != x.rows() || cols() != x.cols()) {
            throw std::invalid_argument(
                "DenseMatrix.axpy() requires matching dtype and shape.");
        }
        ensure_allocated(rxmesh::DEVICE);
        x.ensure_allocated(rxmesh::DEVICE);
        std::visit(
            [&](auto& y_mat) {
                using YMatT = std::decay_t<decltype(y_mat)>;
                std::visit(
                    [&](auto& x_mat) {
                        using XMatT = std::decay_t<decltype(x_mat)>;
                        if constexpr (std::is_same_v<YMatT, XMatT>) {
                            using T = typename YMatT::element_type::Type;
                            if constexpr (std::is_same_v<T, int32_t>) {
                                throw std::invalid_argument(
                                    "DenseMatrix.axpy() supports float32 and "
                                    "float64.");
                            } else {
                                y_mat->axpy(*x_mat, alpha.cast<T>());
                            }
                        }
                    },
                    x.matrix);
            },
            matrix);
    }

    void multiply(py::object scalar)
    {
        ensure_allocated(rxmesh::DEVICE);
        std::visit(
            [&](auto& mat) {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                if constexpr (std::is_same_v<T, int32_t>) {
                    throw std::invalid_argument(
                        "DenseMatrix.multiply() supports float32 and float64.");
                } else {
                    mat->multiply(scalar.cast<T>());
                }
            },
            matrix);
    }

    void swap(PyDenseMatrix& other)
    {
        if (dtype() != other.dtype() || rows() != other.rows() ||
            cols() != other.cols()) {
            throw std::invalid_argument(
                "DenseMatrix.swap() requires matching dtype and shape.");
        }
        ensure_allocated(rxmesh::DEVICE);
        other.ensure_allocated(rxmesh::DEVICE);
        std::visit(
            [&](auto& lhs) {
                using LhsMatT = std::decay_t<decltype(lhs)>;
                std::visit(
                    [&](auto& rhs) {
                        using RhsMatT = std::decay_t<decltype(rhs)>;
                        if constexpr (std::is_same_v<LhsMatT, RhsMatT>) {
                            using T = typename LhsMatT::element_type::Type;
                            if constexpr (std::is_same_v<T, int32_t>) {
                                throw std::invalid_argument(
                                    "DenseMatrix.swap() supports float32 and "
                                    "float64.");
                            } else {
                                lhs->swap(*rhs);
                            }
                        }
                    },
                    other.matrix);
            },
            matrix);
    }

    void reshape(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0 ||
            rows * cols != this->rows() * this->cols()) {
            throw std::invalid_argument(
                "DenseMatrix.reshape() must preserve element count.");
        }
        std::visit([&](const auto& mat) { mat->reshape(rows, cols); }, matrix);
    }

    std::shared_ptr<PyDenseMatrix> col(int column)
    {
        if (column < 0 || column >= cols()) {
            throw std::out_of_range(
                "DenseMatrix.col() column is out of range.");
        }
        return std::visit(
            [&](const auto& mat) -> std::shared_ptr<PyDenseMatrix> {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                DenseMatrixVariant view =
                    std::make_shared<rxmesh::DenseMatrix<T, Eigen::ColMajor>>(
                        mat->col(column));
                auto ret = std::make_shared<PyDenseMatrix>(view, allocated);
                ret->base_owner = shared_from_this();
                return ret;
            },
            matrix);
    }

    std::shared_ptr<PyDenseMatrix> segment(int start, int count)
    {
        if (start < 0 || count < 0 || start + count > rows() * cols()) {
            throw std::out_of_range(
                "DenseMatrix.segment() range is out of bounds.");
        }
        return std::visit(
            [&](const auto& mat) -> std::shared_ptr<PyDenseMatrix> {
                using MatT = std::decay_t<decltype(mat)>;
                using T    = typename MatT::element_type::Type;
                DenseMatrixVariant view =
                    std::make_shared<rxmesh::DenseMatrix<T, Eigen::ColMajor>>(
                        mat->segment(start, count));
                auto ret = std::make_shared<PyDenseMatrix>(view, allocated);
                ret->base_owner = shared_from_this();
                return ret;
            },
            matrix);
    }

    void to_mtx(const std::string& file_name)
    {
        ensure_allocated(rxmesh::HOST);
        std::visit([&](const auto& mat) { mat->to_mtx(file_name); }, matrix);
    }
};

}  // namespace pyrxmesh_py
