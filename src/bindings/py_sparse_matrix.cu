#include "bindings/py_hessian_sparse_matrix.h"
#include "bindings/py_jacobian_sparse_matrix.h"
#include "bindings/sparse_matrix_csr.h"
#include "bindings/sparse_matrix_dlpack.h"

namespace pyrxmesh_py {

using namespace rxmesh;

std::shared_ptr<PySparseMatrix> make_sparse_matrix(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    rxmesh::Op                            op,
    std::string                           dtype)
{
    switch (parse_dtype(dtype)) {
        case DType::Float32:
            return std::make_shared<PySparseMatrixT<float>>(
                std::move(mesh), op, static_cast<int>(rxmesh::LOCATION_ALL));
        case DType::Float64:
            return std::make_shared<PySparseMatrixT<double>>(
                std::move(mesh), op, static_cast<int>(rxmesh::LOCATION_ALL));
        case DType::Int32:
            return std::make_shared<PySparseMatrixT<int32_t>>(
                std::move(mesh), op, static_cast<int>(rxmesh::LOCATION_ALL));
        default:
            throw std::invalid_argument(
                "SparseMatrix supports float32, float64, and int32 values.");
    }
}

py::object make_sparse_matrix_from_mesh(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    rxmesh::Op                            op,
    std::string                           dtype)
{
    return py::cast(make_sparse_matrix(std::move(mesh), op, std::move(dtype)));
}

template <typename T>
void bind_sparse_matrix_type(py::module_& m, const char* name)
{
    py::class_<PySparseMatrixT<T>,
               PySparseMatrix,
               std::shared_ptr<PySparseMatrixT<T>>>(m, name);
}

template <typename T>
void bind_jacobian_sparse_matrix_type(py::module_& m, const char* name)
{
    py::class_<PyJacobianSparseMatrix<T>,
               PySparseMatrixT<T>,
               std::shared_ptr<PyJacobianSparseMatrix<T>>>(m, name)
        .def_property_readonly("num_terms",
                               &PyJacobianSparseMatrix<T>::num_terms)
        .def("term_num_rows",
             &PyJacobianSparseMatrix<T>::term_num_rows,
             py::arg("term"))
        .def("term_rows_range",
             &PyJacobianSparseMatrix<T>::term_rows_range,
             py::arg("term"));
}

template <typename T, int K>
void bind_hessian_sparse_matrix_type(py::module_& m, const char* name)
{
    py::class_<PyHessianSparseMatrix<T, K>,
               PySparseMatrixT<T>,
               std::shared_ptr<PyHessianSparseMatrix<T, K>>>(m, name)
        .def_property_readonly("variable_dim",
                               &PyHessianSparseMatrix<T, K>::variable_dim);
}

py::object make_jacobian_sparse_matrix(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    std::vector<rxmesh::Op>               ops,
    py::sequence                          block_shapes,
    std::string                           dtype)
{
    switch (parse_dtype(dtype)) {
        case DType::Float32:
            return py::cast(std::make_shared<PyJacobianSparseMatrix<float>>(
                std::move(mesh), std::move(ops), block_shapes));
        case DType::Float64:
            return py::cast(std::make_shared<PyJacobianSparseMatrix<double>>(
                std::move(mesh), std::move(ops), block_shapes));
        default:
            throw std::invalid_argument(
                "JacobianSparseMatrix supports float32 and float64.");
    }
}

template <typename T>
py::object make_hessian_sparse_matrix_typed(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    int                                   variable_dim,
    int                                   extra_nnz_entries,
    rxmesh::Op                            op)
{
    switch (variable_dim) {
        case 1:
            return py::cast(std::make_shared<PyHessianSparseMatrix<T, 1>>(
                std::move(mesh), extra_nnz_entries, op));
        case 2:
            return py::cast(std::make_shared<PyHessianSparseMatrix<T, 2>>(
                std::move(mesh), extra_nnz_entries, op));
        case 3:
            return py::cast(std::make_shared<PyHessianSparseMatrix<T, 3>>(
                std::move(mesh), extra_nnz_entries, op));
        case 4:
            return py::cast(std::make_shared<PyHessianSparseMatrix<T, 4>>(
                std::move(mesh), extra_nnz_entries, op));
        case 6:
            return py::cast(std::make_shared<PyHessianSparseMatrix<T, 6>>(
                std::move(mesh), extra_nnz_entries, op));
        default:
            throw std::invalid_argument(
                "HessianSparseMatrix currently supports variable_dim 1, 2, 3, "
                "4, or 6.");
    }
}

py::object make_hessian_sparse_matrix(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    int                                   variable_dim,
    int                                   extra_nnz_entries,
    rxmesh::Op                            op,
    std::string                           dtype)
{
    validate_hessian_inputs(mesh, variable_dim, extra_nnz_entries);
    switch (parse_dtype(dtype)) {
        case DType::Float32:
            return make_hessian_sparse_matrix_typed<float>(
                std::move(mesh), variable_dim, extra_nnz_entries, op);
        case DType::Float64:
            return make_hessian_sparse_matrix_typed<double>(
                std::move(mesh), variable_dim, extra_nnz_entries, op);
        default:
            throw std::invalid_argument(
                "HessianSparseMatrix supports float32 and float64.");
    }
}

void register_sparse_matrix(py::module_& m)
{
    py::class_<PySparseMatrix, std::shared_ptr<PySparseMatrix>>(m,
                                                                "SparseMatrix")
        .def(py::init(&make_sparse_matrix),
             py::arg("mesh"),
             py::arg("op")    = rxmesh::Op::VV,
             py::arg("dtype") = "float32")
        .def_static("from_numpy_copy",
                    &sparse_matrix_from_numpy_copy,
                    py::arg("row_ptr"),
                    py::arg("col_idx"),
                    py::arg("values"),
                    py::arg("shape"),
                    py::arg("dtype") = "float32",
                    "Copy CSR arrays into a new RXMesh-owned SparseMatrix.")
        .def_static("from_dlpack_copy",
                    &sparse_matrix_from_dlpack_copy,
                    py::arg("row_ptr"),
                    py::arg("col_idx"),
                    py::arg("values"),
                    py::arg("shape"),
                    py::arg("dtype")  = "",
                    py::arg("stream") = py::none(),
                    "Copy CSR DLPack tensors into a new RXMesh-owned "
                    "SparseMatrix.")
        .def_property_readonly("rows", &PySparseMatrix::rows)
        .def_property_readonly("cols", &PySparseMatrix::cols)
        .def_property_readonly("shape", &PySparseMatrix::shape)
        .def_property_readonly("nnz", &PySparseMatrix::nnz)
        .def_property_readonly("non_zeros", &PySparseMatrix::nnz)
        .def_property_readonly("lower_nnz", &PySparseMatrix::lower_nnz)
        .def_property_readonly("dtype", &PySparseMatrix::dtype)
        .def_property_readonly("index_dtype", &PySparseMatrix::index_dtype)
        .def_property_readonly(
            "op", [](const PySparseMatrix& self) { return self.op; })
        .def_property_readonly("location", &PySparseMatrix::location)
        .def_property_readonly("is_host_allocated",
                               &PySparseMatrix::is_host_allocated)
        .def_property_readonly("is_device_allocated",
                               &PySparseMatrix::is_device_allocated)
        .def("move",
             &PySparseMatrix::move,
             py::arg("source"),
             py::arg("target"),
             py::arg("stream") = py::none())
        .def("sync_host_to_device",
             &PySparseMatrix::sync_host_to_device,
             py::arg("stream") = py::none())
        .def("sync_device_to_host",
             &PySparseMatrix::sync_device_to_host,
             py::arg("stream") = py::none())
        .def("reset",
             &PySparseMatrix::reset,
             py::arg("value"),
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("stream")   = py::none())
        .def("copy_from",
             &PySparseMatrix::copy_from,
             py::arg("other"),
             py::arg("source") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("stream") = py::none())
        .def("is_non_zero",
             &PySparseMatrix::is_non_zero,
             py::arg("row"),
             py::arg("col"))
        .def("value", &PySparseMatrix::value, py::arg("row"), py::arg("col"))
        .def("set_value",
             &PySparseMatrix::set_value,
             py::arg("row"),
             py::arg("col"),
             py::arg("value"))
        .def("to_numpy",
             &PySparseMatrix::to_numpy,
             py::arg("location") = static_cast<int>(rxmesh::HOST))
        .def(
            "to_numpy_copy",
            [](PySparseMatrix& self, int source, py::object) {
                return self.to_numpy_copy(source);
            },
            py::arg("source") = static_cast<int>(rxmesh::HOST),
            py::arg("stream") = py::none())
        .def("values_to_numpy",
             &PySparseMatrix::values_to_numpy,
             py::arg("location") = static_cast<int>(rxmesh::HOST))
        .def(
            "values_to_numpy_copy",
            [](PySparseMatrix& self, int source, py::object) {
                return self.values_to_numpy_copy(source);
            },
            py::arg("source") = static_cast<int>(rxmesh::HOST),
            py::arg("stream") = py::none())
        .def("from_numpy_values_copy",
             &PySparseMatrix::from_numpy_values_copy,
             py::arg("values"),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("stream") = py::none())
        .def("from_dlpack_values_copy",
             &sparse_values_from_dlpack_copy,
             py::arg("values"),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("stream") = py::none())
        .def("multiply",
             &PySparseMatrix::multiply_dense,
             py::arg("rhs"),
             py::arg("transpose_a") = false,
             py::arg("transpose_b") = false,
             py::arg("alpha")       = py::float_(1.0),
             py::arg("beta")        = py::float_(0.0),
             py::arg("stream")      = py::none())
        .def("multiply_vector",
             &PySparseMatrix::multiply_vector,
             py::arg("rhs"),
             py::arg("stream") = py::none())
        .def("to_mtx", &PySparseMatrix::to_mtx, py::arg("file_name"))
        .def("to_file", &PySparseMatrix::to_file, py::arg("file_name"))
        .def("release", &PySparseMatrix::release)
        .def(
            "to_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               int                             location,
               py::object                      stream) {
                auto row_ptr = sparse_component_to_dlpack(
                    self, CsrComponent::RowPtr, location, stream);
                auto col_idx = sparse_component_to_dlpack(
                    self, CsrComponent::ColIdx, location, stream);
                auto values = sparse_component_to_dlpack(std::move(self),
                                                         CsrComponent::Values,
                                                         location,
                                                         std::move(stream));
                return py::make_tuple(
                    std::move(row_ptr), std::move(col_idx), std::move(values));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none(),
            "Return DLPack capsules for (row_ptr, col_indices, values).")
        .def(
            "_row_ptr_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               int                             location,
               py::object                      stream) {
                return sparse_component_to_dlpack(std::move(self),
                                                  CsrComponent::RowPtr,
                                                  location,
                                                  std::move(stream));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none())
        .def(
            "_crow_indices_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               int                             location,
               py::object                      stream) {
                return sparse_component_to_dlpack(std::move(self),
                                                  CsrComponent::RowPtr,
                                                  location,
                                                  std::move(stream));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none())
        .def(
            "_col_indices_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               int                             location,
               py::object                      stream) {
                return sparse_component_to_dlpack(std::move(self),
                                                  CsrComponent::ColIdx,
                                                  location,
                                                  std::move(stream));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none())
        .def(
            "_col_idx_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               int                             location,
               py::object                      stream) {
                return sparse_component_to_dlpack(std::move(self),
                                                  CsrComponent::ColIdx,
                                                  location,
                                                  std::move(stream));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none())
        .def(
            "_values_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               int                             location,
               py::object                      stream) {
                return sparse_component_to_dlpack(std::move(self),
                                                  CsrComponent::Values,
                                                  location,
                                                  std::move(stream));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none());

    bind_sparse_matrix_type<float>(m, "_SparseMatrixFloat32");
    bind_sparse_matrix_type<double>(m, "_SparseMatrixFloat64");
    bind_sparse_matrix_type<int32_t>(m, "_SparseMatrixInt32");

    bind_jacobian_sparse_matrix_type<float>(m, "_JacobianSparseMatrixFloat32");
    bind_jacobian_sparse_matrix_type<double>(m, "_JacobianSparseMatrixFloat64");
    m.def("JacobianSparseMatrix",
          &make_jacobian_sparse_matrix,
          py::arg("mesh"),
          py::arg("ops"),
          py::arg("block_shapes"),
          py::arg("dtype") = "float32");

    bind_hessian_sparse_matrix_type<float, 1>(
        m, "_HessianSparseMatrixFloat32Dim1");
    bind_hessian_sparse_matrix_type<float, 2>(
        m, "_HessianSparseMatrixFloat32Dim2");
    bind_hessian_sparse_matrix_type<float, 3>(
        m, "_HessianSparseMatrixFloat32Dim3");
    bind_hessian_sparse_matrix_type<float, 4>(
        m, "_HessianSparseMatrixFloat32Dim4");
    bind_hessian_sparse_matrix_type<float, 6>(
        m, "_HessianSparseMatrixFloat32Dim6");
    bind_hessian_sparse_matrix_type<double, 1>(
        m, "_HessianSparseMatrixFloat64Dim1");
    bind_hessian_sparse_matrix_type<double, 2>(
        m, "_HessianSparseMatrixFloat64Dim2");
    bind_hessian_sparse_matrix_type<double, 3>(
        m, "_HessianSparseMatrixFloat64Dim3");
    bind_hessian_sparse_matrix_type<double, 4>(
        m, "_HessianSparseMatrixFloat64Dim4");
    bind_hessian_sparse_matrix_type<double, 6>(
        m, "_HessianSparseMatrixFloat64Dim6");
    m.def("HessianSparseMatrix",
          &make_hessian_sparse_matrix,
          py::arg("mesh"),
          py::arg("variable_dim")      = 3,
          py::arg("extra_nnz_entries") = 0,
          py::arg("op")                = rxmesh::Op::VV,
          py::arg("dtype")             = "float32");
}

}  // namespace pyrxmesh_py
