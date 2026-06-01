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

std::shared_ptr<PyDenseMatrix> make_dense_matrix_for_mesh(
    std::shared_ptr<rxmesh::RXMeshStatic> mesh,
    int                                   rows,
    int                                   cols,
    std::string                           dtype,
    int                                   location,
    std::string                           order)
{
    return std::make_shared<PyDenseMatrix>(
        std::move(mesh), dtype, rows, cols, location, std::move(order));
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
        .def(py::init(&make_dense_matrix_for_mesh),
             py::arg("mesh"),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("dtype")    = "float32",
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("order")    = "col_major")
        .def_static("from_dlpack_copy",
                    &dense_matrix_from_dlpack_copy,
                    py::arg("source"),
                    "Copy a 2D CPU or CUDA DLPack tensor into new RXMesh "
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
        .def(
            "move",
            [](PyDenseMatrix& self, int source, int target, py::object stream) {
                self.move(parse_location(source),
                          parse_location(target),
                          parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("source"),
            py::arg("target"),
            py::arg("stream") = py::none())
        .def("release",
             &PyDenseMatrix::release,
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def(
            "reset",
            [](PyDenseMatrix& self,
               py::object     value,
               int            location,
               py::object     stream) {
                self.reset(std::move(value),
                           location,
                           parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("value"),
            py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL),
            py::arg("stream")   = py::none())
        .def("fill_random",
             &PyDenseMatrix::fill_random,
             py::arg("low")  = -1.0,
             py::arg("high") = 1.0)
        .def("value",
             &PyDenseMatrix::value,
             py::arg("row_or_handle"),
             py::arg("col") = 0)
        .def("set_value",
             &PyDenseMatrix::set_value,
             py::arg("row_or_handle"),
             py::arg("col"),
             py::arg("value"))
        .def("to_numpy",
             &PyDenseMatrix::to_numpy,
             py::arg("location") = static_cast<int>(rxmesh::HOST))
        .def("to_numpy_copy",
             &PyDenseMatrix::to_numpy_copy,
             py::arg("source") = static_cast<int>(rxmesh::HOST))
        .def(
            "from_numpy_copy",
            [](PyDenseMatrix& self,
               py::array      values,
               int            target,
               py::object     stream) {
                self.from_numpy_copy(std::move(values),
                                     target,
                                     parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("values"),
            py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
            py::arg("stream") = py::none())
        .def(
            "copy_from",
            [](PyDenseMatrix& self,
               PyDenseMatrix& other,
               int            source,
               int            target,
               py::object     stream) {
                self.copy_from(other,
                               source,
                               target,
                               parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("other"),
            py::arg("source") = static_cast<int>(rxmesh::LOCATION_ALL),
            py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
            py::arg("stream") = py::none())
        .def(
            "norm2",
            [](PyDenseMatrix& self, py::object stream) {
                return self.norm2(parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("stream") = py::none())
        .def(
            "abs_sum",
            [](PyDenseMatrix& self, py::object stream) {
                return self.abs_sum(parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("stream") = py::none())
        .def(
            "abs_max",
            [](PyDenseMatrix& self, py::object stream) {
                return self.abs_max(parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("stream") = py::none())
        .def(
            "abs_min",
            [](PyDenseMatrix& self, py::object stream) {
                return self.abs_min(parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("stream") = py::none())
        .def(
            "dot",
            [](PyDenseMatrix& self, PyDenseMatrix& other, py::object stream) {
                return self.dot(other,
                                parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("other"),
            py::arg("stream") = py::none())
        .def(
            "axpy",
            [](PyDenseMatrix& self,
               PyDenseMatrix& x,
               py::object     alpha,
               py::object     stream) {
                self.axpy(x,
                          std::move(alpha),
                          parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("x"),
            py::arg("alpha"),
            py::arg("stream") = py::none())
        .def(
            "multiply",
            [](PyDenseMatrix& self, py::object scalar, py::object stream) {
                self.multiply(std::move(scalar),
                              parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("scalar"),
            py::arg("stream") = py::none())
        .def(
            "swap",
            [](PyDenseMatrix& self, PyDenseMatrix& other, py::object stream) {
                self.swap(other, parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("other"),
            py::arg("stream") = py::none())
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
