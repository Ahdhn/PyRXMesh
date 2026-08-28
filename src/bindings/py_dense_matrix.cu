#include "bindings/dense_matrix_dlpack.h"
#include "bindings/py_dense_matrix.h"

namespace pyrxmesh_py {

void register_dense_matrix(py::module_& m)
{
    py::class_<PyDenseMatrix, std::shared_ptr<PyDenseMatrix>>(m, "DenseMatrix")
        .def(py::init([](int                rows,
                         int                cols,
                         const std::string& dtype,
                         py::object         location,
                         const std::string& order) {
                 return make_dense_matrix(
                     rows,
                     cols,
                     dtype,
                     static_cast<int>(parse_location(location)),
                     order);
             }),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("dtype")    = "float32",
             py::arg("location") = "all",
             py::arg("order")    = "col_major")
        .def(py::init([](std::shared_ptr<rxmesh::RXMeshStatic> mesh,
                         int                                   rows,
                         int                                   cols,
                         const std::string&                    dtype,
                         py::object                            location,
                         const std::string&                    order) {
                 return make_dense_matrix_for_mesh(
                     std::move(mesh),
                     rows,
                     cols,
                     dtype,
                     static_cast<int>(parse_location(location)),
                     order);
             }),
             py::arg("mesh"),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("dtype")    = "float32",
             py::arg("location") = "all",
             py::arg("order")    = "col_major")
        .def_static("from_dlpack_copy",
                    &dense_matrix_from_dlpack_copy,
                    py::arg("source"),
                    py::arg("order") = "col_major",
                    "Copy a 2D CPU or CUDA DLPack tensor into new RXMesh "
                    "DenseMatrix memory.")
        .def_static("from_dlpack_view",
                    &dense_matrix_from_dlpack_view,
                    py::arg("source"),
                    py::arg("order") = "col_major",
                    "View a compact 2D CPU or CUDA DLPack tensor as a "
                    "DenseMatrix without taking ownership of its memory.")
        .def_property_readonly("rows", &PyDenseMatrix::rows)
        .def_property_readonly("cols", &PyDenseMatrix::cols)
        .def_property_readonly("shape", &PyDenseMatrix::shape)
        .def_property_readonly("dtype", &PyDenseMatrix::dtype)
        .def_property_readonly("order", &PyDenseMatrix::order)
        .def_property_readonly(
            "location",
            [](const PyDenseMatrix& self) {
                return location_name(parse_location(self.location()));
            })
        .def_property_readonly("bytes", &PyDenseMatrix::bytes)
        .def_property_readonly("is_host_allocated",
                               &PyDenseMatrix::is_host_allocated)
        .def_property_readonly("is_device_allocated",
                               &PyDenseMatrix::is_device_allocated)
        .def_property_readonly("is_view", &PyDenseMatrix::is_view)
        .def_property_readonly("is_read_only", &PyDenseMatrix::is_read_only)
        .def(
            "move",
            [](PyDenseMatrix& self,
               py::object     source,
               py::object     target,
               py::object     stream) {
                self.move(parse_location(source),
                          parse_location(target),
                          parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("source"),
            py::arg("target"),
            py::arg("stream") = py::none())
        .def(
            "release",
            [](PyDenseMatrix& self, py::object location) {
                self.release(static_cast<int>(parse_location(location)));
            },
            py::arg("location") = "all")
        .def(
            "reset",
            [](PyDenseMatrix& self,
               py::object     value,
               py::object     location,
               py::object     stream) {
                self.reset(std::move(value),
                           static_cast<int>(parse_location(location)),
                           parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("value"),
            py::arg("location") = "all",
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
        .def(
            "to_numpy",
            [](PyDenseMatrix& self, py::object location) {
                return self.to_numpy(
                    static_cast<int>(parse_location(location)));
            },
            py::arg("location") = "host")
        .def(
            "to_numpy_copy",
            [](PyDenseMatrix& self, py::object source) {
                return self.to_numpy_copy(
                    static_cast<int>(parse_location(source)));
            },
            py::arg("source") = "host")
        .def(
            "from_numpy_copy",
            [](PyDenseMatrix& self,
               py::array      values,
               py::object     target,
               py::object     stream) {
                self.from_numpy_copy(std::move(values),
                                     static_cast<int>(parse_location(target)),
                                     parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("values"),
            py::arg("target") = "all",
            py::arg("stream") = py::none())
        .def(
            "copy_from",
            [](PyDenseMatrix& self,
               PyDenseMatrix& other,
               py::object     source,
               py::object     target,
               py::object     stream) {
                self.copy_from(other,
                               static_cast<int>(parse_location(source)),
                               static_cast<int>(parse_location(target)),
                               parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("other"),
            py::arg("source") = "all",
            py::arg("target") = "all",
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
               py::object                     location,
               py::object                     stream) {
                return dense_matrix_to_dlpack(
                    std::move(self),
                    static_cast<int>(parse_location(location)),
                    std::move(stream));
            },
            py::arg("location") = "device",
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
