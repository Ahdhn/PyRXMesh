#include <sstream>
#include <string>

#include <pybind11/pybind11.h>

#include "rxmesh/handle.h"

namespace py = pybind11;

namespace pyrxmesh_py {

template <typename HandleT>
inline std::string handle_repr(const char* class_name, const HandleT& handle)
{
    std::ostringstream out;
    out << class_name << "(unique_id=" << handle.unique_id()
        << ", patch_id=" << handle.patch_id()
        << ", local_id=" << handle.local_id() << ")";
    return out.str();
}

template <typename HandleT>
void bind_basic_handle(py::module_& m, const char* class_name)
{
    py::class_<HandleT>(m, class_name)
        .def(py::init<>())
        .def(py::init<uint64_t>(), py::arg("unique_id"))
        .def(py::init([](uint32_t patch_id, uint16_t local_id) {
                 return HandleT(patch_id, typename HandleT::LocalT(local_id));
             }),
             py::arg("patch_id"),
             py::arg("local_id"))
        .def_property_readonly("unique_id", &HandleT::unique_id)
        .def_property_readonly("patch_id", &HandleT::patch_id)
        .def_property_readonly("local_id", &HandleT::local_id)
        .def("is_valid", &HandleT::is_valid)
        .def("__int__", &HandleT::unique_id)
        .def("__index__", &HandleT::unique_id)
        .def("__hash__",
             [](const HandleT& self) {
                 return std::hash<uint64_t>{}(self.unique_id());
             })
        .def("__eq__",
             [](const HandleT& self, const HandleT& other) {
                 return self == other;
             })
        .def("__ne__",
             [](const HandleT& self, const HandleT& other) {
                 return self != other;
             })
        .def("__repr__", [class_name](const HandleT& self) {
            return handle_repr(class_name, self);
        });
}

void register_handles(py::module_& m)
{
    bind_basic_handle<rxmesh::VertexHandle>(m, "VertexHandle");
    bind_basic_handle<rxmesh::EdgeHandle>(m, "EdgeHandle");
    bind_basic_handle<rxmesh::FaceHandle>(m, "FaceHandle");

    py::class_<rxmesh::DEdgeHandle>(m, "DEdgeHandle")
        .def(py::init<>())
        .def(py::init([](uint32_t       patch_id,
                         uint16_t       local_id,
                         rxmesh::flag_t direction) {
                 return rxmesh::DEdgeHandle(
                     patch_id, rxmesh::LocalEdgeT(local_id), direction);
             }),
             py::arg("patch_id"),
             py::arg("local_id"),
             py::arg("direction"))
        .def_property_readonly("unique_id", &rxmesh::DEdgeHandle::unique_id)
        .def_property_readonly("patch_id", &rxmesh::DEdgeHandle::patch_id)
        .def_property_readonly("local_id", &rxmesh::DEdgeHandle::local_id)
        .def("is_valid", &rxmesh::DEdgeHandle::is_valid)
        .def("flipped", &rxmesh::DEdgeHandle::get_flip_dedge)
        .def("edge_handle", &rxmesh::DEdgeHandle::get_edge_handle)
        .def("__int__", &rxmesh::DEdgeHandle::unique_id)
        .def("__index__", &rxmesh::DEdgeHandle::unique_id)
        .def("__hash__",
             [](const rxmesh::DEdgeHandle& self) {
                 return std::hash<uint64_t>{}(self.unique_id());
             })
        .def("__eq__",
             [](const rxmesh::DEdgeHandle& self,
                const rxmesh::DEdgeHandle& other) { return self == other; })
        .def("__ne__",
             [](const rxmesh::DEdgeHandle& self,
                const rxmesh::DEdgeHandle& other) { return self != other; })
        .def("__repr__", [](const rxmesh::DEdgeHandle& self) {
            return handle_repr("DEdgeHandle", self);
        });
}

}  // namespace pyrxmesh_py
