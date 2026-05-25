#include <pybind11/pybind11.h>

#include "bindings/common.h"

namespace py = pybind11;

PYBIND11_MODULE(_rxmesh, m)
{
    m.doc() = "Python bindings for RXMesh";

    pyrxmesh_py::register_module_core(m);
    pyrxmesh_py::register_handles(m);
    pyrxmesh_py::register_attribute(m);
    pyrxmesh_py::register_dense_matrix(m);
    pyrxmesh_py::register_geometry(m);
    pyrxmesh_py::register_sparse_matrix(m);
    pyrxmesh_py::register_solvers(m);
    pyrxmesh_py::register_mesh(m);
}
