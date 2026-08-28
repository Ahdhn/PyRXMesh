#include "bindings/dispatch.h"
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
    return dispatch_numeric_dtype_str(
        dtype, [&](auto tag) -> std::shared_ptr<PySparseMatrix> {
            using T = typename decltype(tag)::type;
            return std::make_shared<PySparseMatrixT<T>>(
                std::move(mesh), op, static_cast<int>(rxmesh::LOCATION_ALL));
        });
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
    py::sequence                          ops,
    py::sequence                          block_shapes,
    std::string                           dtype)
{
    std::vector<rxmesh::Op> parsed_ops;
    parsed_ops.reserve(static_cast<size_t>(py::len(ops)));
    for (py::handle op : ops) {
        parsed_ops.push_back(parse_op(op));
    }
    return dispatch_float_dtype_str(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        return py::cast(std::make_shared<PyJacobianSparseMatrix<T>>(
            std::move(mesh), std::move(parsed_ops), block_shapes));
    });
}

py::object make_hessian_sparse_matrix(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    int                                   variable_dim,
    int                                   extra_nnz_entries,
    py::object                            op,
    std::string                           dtype)
{
    validate_hessian_inputs(mesh, variable_dim, extra_nnz_entries);
    const auto parsed_op = parse_op(op);
    return dispatch_float_dtype_str(dtype, [&](auto dtype_tag_v) {
        using T = typename decltype(dtype_tag_v)::type;
        return dispatch_int<1, 2, 3, 4, 6>(variable_dim, [&](auto k_tag) {
            constexpr int K = decltype(k_tag)::value;
            return py::cast(std::make_shared<PyHessianSparseMatrix<T, K>>(
                std::move(mesh), extra_nnz_entries, parsed_op));
        });
    });
}

void register_sparse_matrix(py::module_& m)
{
    py::class_<PySparseMatrix, std::shared_ptr<PySparseMatrix>>(m,
                                                                "SparseMatrix")
        .def(py::init([](std::shared_ptr<rxmesh::RXMeshStatic> mesh,
                         py::object                            op,
                         std::string                           dtype) {
                 return make_sparse_matrix(
                     std::move(mesh), parse_op(op), std::move(dtype));
             }),
             py::arg("mesh"),
             py::arg("op")    = "vv",
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
        .def_property_readonly("lower_nnz", &PySparseMatrix::lower_nnz)
        .def_property_readonly("dtype", &PySparseMatrix::dtype)
        .def_property_readonly("index_dtype", &PySparseMatrix::index_dtype)
        .def_property_readonly(
            "op", [](const PySparseMatrix& self) { return op_name(self.op); })
        .def_property_readonly(
            "location",
            [](const PySparseMatrix& self) {
                return location_name(parse_location(self.location()));
            })
        .def_property_readonly("is_host_allocated",
                               &PySparseMatrix::is_host_allocated)
        .def_property_readonly("is_device_allocated",
                               &PySparseMatrix::is_device_allocated)
        .def(
            "move",
            [](PySparseMatrix& self,
               py::object      source,
               py::object      target,
               py::object      stream) {
                self.move(static_cast<int>(parse_location(source)),
                          static_cast<int>(parse_location(target)),
                          std::move(stream));
            },
            py::arg("source"),
            py::arg("target"),
            py::arg("stream") = py::none())
        .def("sync_host_to_device",
             &PySparseMatrix::sync_host_to_device,
             py::arg("stream") = py::none())
        .def("sync_device_to_host",
             &PySparseMatrix::sync_device_to_host,
             py::arg("stream") = py::none())
        .def(
            "reset",
            [](PySparseMatrix& self,
               py::object      value,
               py::object      location,
               py::object      stream) {
                self.reset(std::move(value),
                           static_cast<int>(parse_location(location)),
                           std::move(stream));
            },
            py::arg("value"),
            py::arg("location") = "all",
            py::arg("stream")   = py::none())
        .def(
            "copy_from",
            [](PySparseMatrix& self,
               PySparseMatrix& other,
               py::object      source,
               py::object      target,
               py::object      stream) {
                self.copy_from(other,
                               static_cast<int>(parse_location(source)),
                               static_cast<int>(parse_location(target)),
                               std::move(stream));
            },
            py::arg("other"),
            py::arg("source") = "all",
            py::arg("target") = "all",
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
        .def(
            "to_numpy",
            [](PySparseMatrix& self, py::object location) {
                return self.to_numpy(
                    static_cast<int>(parse_location(location)));
            },
            py::arg("location") = "host")
        .def(
            "to_numpy_copy",
            [](PySparseMatrix& self, py::object source, py::object) {
                return self.to_numpy_copy(
                    static_cast<int>(parse_location(source)));
            },
            py::arg("source") = "host",
            py::arg("stream") = py::none())
        .def(
            "values_to_numpy",
            [](PySparseMatrix& self, py::object location) {
                return self.values_to_numpy(
                    static_cast<int>(parse_location(location)));
            },
            py::arg("location") = "host")
        .def(
            "values_to_numpy_copy",
            [](PySparseMatrix& self, py::object source, py::object) {
                return self.values_to_numpy_copy(
                    static_cast<int>(parse_location(source)));
            },
            py::arg("source") = "host",
            py::arg("stream") = py::none())
        .def(
            "from_numpy_values_copy",
            [](PySparseMatrix& self,
               py::array       values,
               py::object      target,
               py::object      stream) {
                self.from_numpy_values_copy(
                    std::move(values),
                    static_cast<int>(parse_location(target)),
                    std::move(stream));
            },
            py::arg("values"),
            py::arg("target") = "all",
            py::arg("stream") = py::none())
        .def(
            "from_dlpack_values_copy",
            [](PySparseMatrix& self,
               py::object      values,
               py::object      target,
               py::object      stream) {
                sparse_values_from_dlpack_copy(
                    self,
                    std::move(values),
                    static_cast<int>(parse_location(target)),
                    std::move(stream));
            },
            py::arg("values"),
            py::arg("target") = "all",
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
               py::object                      location,
               py::object                      stream) {
                const int loc     = static_cast<int>(parse_location(location));
                auto      row_ptr = sparse_component_to_dlpack(
                    self, CsrComponent::RowPtr, loc, stream);
                auto col_idx = sparse_component_to_dlpack(
                    self, CsrComponent::ColIdx, loc, stream);
                auto values = sparse_component_to_dlpack(std::move(self),
                                                         CsrComponent::Values,
                                                         loc,
                                                         std::move(stream));
                return py::make_tuple(
                    std::move(row_ptr), std::move(col_idx), std::move(values));
            },
            py::arg("location") = "device",
            py::arg("stream")   = py::none(),
            "Return DLPack capsules for (row_ptr, col_indices, values).")
        .def(
            "_row_ptr_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               py::object                      location,
               py::object                      stream) {
                return sparse_component_to_dlpack(
                    std::move(self),
                    CsrComponent::RowPtr,
                    static_cast<int>(parse_location(location)),
                    std::move(stream));
            },
            py::arg("location") = "device",
            py::arg("stream")   = py::none())
        .def(
            "_col_indices_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               py::object                      location,
               py::object                      stream) {
                return sparse_component_to_dlpack(
                    std::move(self),
                    CsrComponent::ColIdx,
                    static_cast<int>(parse_location(location)),
                    std::move(stream));
            },
            py::arg("location") = "device",
            py::arg("stream")   = py::none())
        .def(
            "_values_dlpack",
            [](std::shared_ptr<PySparseMatrix> self,
               py::object                      location,
               py::object                      stream) {
                return sparse_component_to_dlpack(
                    std::move(self),
                    CsrComponent::Values,
                    static_cast<int>(parse_location(location)),
                    std::move(stream));
            },
            py::arg("location") = "device",
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
          py::arg("op")                = "vv",
          py::arg("dtype")             = "float32");
}

}  // namespace pyrxmesh_py
