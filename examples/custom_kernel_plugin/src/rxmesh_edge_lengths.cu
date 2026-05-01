#include <pybind11/pybind11.h>

#include "pyrxmesh/plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;

void compute_edge_lengths(py::object mesh_obj,
                          py::object coords_obj,
                          py::object out_obj)
{
    auto coords = pyrxmesh::vertex_attribute<float>(coords_obj);
    auto out    = pyrxmesh::edge_attribute<float>(out_obj);

    if (coords.get_num_attributes() < 3) {
        throw std::runtime_error("coords must have at least 3 values per vertex.");
    }
    if (out.get_num_attributes() != 1) {
        throw std::runtime_error("out must have exactly 1 value per edge.");
    }

    pyrxmesh::for_each<Op::EV, 256>(
        mesh_obj,
        [coords, out] __device__(const EdgeHandle& eh,
                                 const VertexIterator& iter) mutable {
            const Eigen::Vector3f a = coords.to_eigen<3>(iter[0]);
            const Eigen::Vector3f b = coords.to_eigen<3>(iter[1]);
            out(eh) = (a - b).norm();
        });
}

PYBIND11_MODULE(_rxmesh_edge_lengths, m)
{
    pyrxmesh::require_compatible_runtime(m);

    m.def("compute_edge_lengths",
          &compute_edge_lengths,
          py::arg("mesh"),
          py::arg("coords"),
          py::arg("out"));
}
