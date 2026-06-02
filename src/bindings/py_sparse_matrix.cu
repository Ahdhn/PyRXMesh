#include "bindings/py_sparse_diff.h"
#include "bindings/sparse_matrix_csr.h"
#include "bindings/sparse_matrix_dlpack.h"

namespace pyrxmesh_py {

using namespace rxmesh;

std::shared_ptr<PySparseMatrix> make_sparse_matrix(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    rxmesh::Op                            op,
    std::string                           dtype)
{
    return std::make_shared<PySparseMatrix>(std::move(mesh), op, dtype);
}

py::object make_sparse_matrix_from_mesh(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    rxmesh::Op                            op,
    std::string                           dtype)
{
    return py::cast(make_sparse_matrix(std::move(mesh), op, dtype));
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
        .def("to_numpy_copy",
             &PySparseMatrix::to_numpy_copy,
             py::arg("source") = static_cast<int>(rxmesh::HOST),
             py::arg("stream") = py::none())
        .def("values_to_numpy",
             &PySparseMatrix::values_to_numpy,
             py::arg("location") = static_cast<int>(rxmesh::HOST))
        .def("values_to_numpy_copy",
             &PySparseMatrix::values_to_numpy_copy,
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
        .def("transpose",
             &PySparseMatrix::transpose,
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
            py::arg("stream")   = py::none(),
            "Return a DLPack capsule for the CSR row pointer array.")
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
            py::arg("stream")   = py::none(),
            "Return a DLPack capsule for the CSR row pointer array.")
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
            py::arg("stream")   = py::none(),
            "Return a DLPack capsule for the CSR column index array.")
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
            py::arg("stream")   = py::none(),
            "Return a DLPack capsule for the CSR column index array.")
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
            py::arg("stream")   = py::none(),
            "Return a DLPack capsule for the CSR value array.");

    py::class_<PyJacobianSparseMatrix,
               PySparseMatrix,
               std::shared_ptr<PyJacobianSparseMatrix>>(m,
                                                        "JacobianSparseMatrix")
        .def(py::init<std::shared_ptr<rxmesh::RXMeshStatic>,
                      std::vector<rxmesh::Op>,
                      py::sequence,
                      std::string>(),
             py::arg("mesh"),
             py::arg("ops"),
             py::arg("block_shapes"),
             py::arg("dtype") = "float32")
        .def_property_readonly("num_terms", &PyJacobianSparseMatrix::num_terms)
        .def("term_num_rows",
             &PyJacobianSparseMatrix::term_num_rows,
             py::arg("term"))
        .def("term_rows_range",
             &PyJacobianSparseMatrix::term_rows_range,
             py::arg("term"));

    py::class_<PyHessianSparseMatrix,
               PySparseMatrix,
               std::shared_ptr<PyHessianSparseMatrix>>(m, "HessianSparseMatrix")
        .def(py::init<std::shared_ptr<rxmesh::RXMeshStatic>,
                      int,
                      int,
                      rxmesh::Op,
                      std::string>(),
             py::arg("mesh"),
             py::arg("variable_dim")      = 3,
             py::arg("extra_nnz_entries") = 0,
             py::arg("op")                = rxmesh::Op::VV,
             py::arg("dtype")             = "float32")
        .def_property_readonly("variable_dim",
                               &PyHessianSparseMatrix::variable_dim);

    py::class_<PyCandidatePairs, std::shared_ptr<PyCandidatePairs>>(
        m, "CandidatePairs")
        .def(py::init<int>(), py::arg("capacity"))
        .def_property_readonly("capacity", &PyCandidatePairs::capacity)
        .def_property_readonly("num_pairs", &PyCandidatePairs::num_pairs)
        .def("insert",
             &PyCandidatePairs::insert,
             py::arg("first"),
             py::arg("second"))
        .def("get_pair", &PyCandidatePairs::get_pair, py::arg("index"))
        .def("to_numpy", &PyCandidatePairs::to_numpy)
        .def("reset", &PyCandidatePairs::reset);
}

}  // namespace pyrxmesh_py
