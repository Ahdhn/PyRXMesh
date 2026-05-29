#include "bindings/dense_matrix_dlpack.h"
#include "bindings/py_dense_matrix.h"

namespace pyrxmesh_py {

namespace {

std::shared_ptr<PyDenseMatrix> make_dense_matrix(int         rows,
                                                 int         cols,
                                                 std::string dtype,
                                                 int         location,
                                                 std::string order)
{
    return std::make_shared<PyDenseMatrix>(
        dtype, rows, cols, location, std::move(order));
}

}  // namespace

void register_dense_matrix(py::module_& m)
{
    py::class_<PyDenseMatrix, std::shared_ptr<PyDenseMatrix>>(m, "DenseMatrix")
        .def(py::init(&make_dense_matrix),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("dtype")    = "float32",
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("order")    = "col_major")
        .def_static("from_dlpack_copy",
                    &dense_matrix_from_dlpack_copy,
                    py::arg("source"),
                    "Copy a 2D CPU or CUDA DLPack tensor into new RXMesh-owned "
                    "DenseMatrix memory.")
        .def_property_readonly("rows", &PyDenseMatrix::rows)
        .def_property_readonly("cols", &PyDenseMatrix::cols)
        .def_property_readonly("shape", &PyDenseMatrix::shape)
        .def_property_readonly("dtype", &PyDenseMatrix::dtype)
        .def_property_readonly("order", &PyDenseMatrix::order)
        .def_property_readonly("location", &PyDenseMatrix::location)
        .def_property_readonly("bytes", &PyDenseMatrix::bytes)
        .def_property_readonly("is_host_allocated",
                               &PyDenseMatrix::is_host_allocated)
        .def_property_readonly("is_device_allocated",
                               &PyDenseMatrix::is_device_allocated)
        .def("move",
             py::overload_cast<int, int>(&PyDenseMatrix::move),
             py::arg("source"),
             py::arg("target"))
        .def("release",
             &PyDenseMatrix::release,
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def("reset",
             &PyDenseMatrix::reset,
             py::arg("value"),
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def("fill_random",
             &PyDenseMatrix::fill_random,
             py::arg("low")  = -1.0,
             py::arg("high") = 1.0)
        .def("to_numpy",
             &PyDenseMatrix::to_numpy,
             py::arg("source") = static_cast<int>(rxmesh::HOST),
             py::arg("copy")   = true)
        .def("from_numpy",
             &PyDenseMatrix::from_numpy,
             py::arg("values"),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def("copy_from",
             &PyDenseMatrix::copy_from,
             py::arg("other"),
             py::arg("source") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def("norm2", &PyDenseMatrix::norm2)
        .def("abs_sum", &PyDenseMatrix::abs_sum)
        .def("abs_max", &PyDenseMatrix::abs_max)
        .def("abs_min", &PyDenseMatrix::abs_min)
        .def("dot", &PyDenseMatrix::dot, py::arg("other"))
        .def("axpy", &PyDenseMatrix::axpy, py::arg("x"), py::arg("alpha"))
        .def("multiply", &PyDenseMatrix::multiply, py::arg("scalar"))
        .def("swap", &PyDenseMatrix::swap, py::arg("other"))
        .def("reshape",
             &PyDenseMatrix::reshape,
             py::arg("rows"),
             py::arg("cols"))
        .def("col", &PyDenseMatrix::col, py::arg("column"))
        .def("segment",
             &PyDenseMatrix::segment,
             py::arg("start"),
             py::arg("count"))
        .def("to_mtx", &PyDenseMatrix::to_mtx, py::arg("file_name"))
        .def(
            "to_dlpack",
            [](std::shared_ptr<PyDenseMatrix> self,
               int                            location,
               py::object                     stream) {
                return dense_matrix_to_dlpack(
                    std::move(self), location, std::move(stream));
            },
            py::arg("location") = static_cast<int>(rxmesh::DEVICE),
            py::arg("stream")   = py::none(),
            "Return a DLPack capsule that views RXMesh-owned memory.")
        .def(
            "__dlpack__",
            [](std::shared_ptr<PyDenseMatrix> self, py::object stream) {
                return dense_matrix_dunder_dlpack(std::move(self),
                                                  std::move(stream));
            },
            py::arg("stream") = py::none())
        .def("__dlpack_device__", [](const PyDenseMatrix& self) {
            return dense_matrix_dlpack_device(self);
        });
}

}  // namespace pyrxmesh_py
