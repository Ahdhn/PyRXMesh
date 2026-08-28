#include <pybind11/pybind11.h>

#include <utility>

#include "pyrxmesh/diff_plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;

using EdgeQuadraticProblem =
    pyrxmesh::diff::ScalarGradientProblem<float, 3, VertexHandle>;

void add_edge_quadratic_terms(EdgeQuadraticProblem& problem, py::dict params)
{
    const float w0 = params.contains("w0") ? params["w0"].cast<float>() : 0.5f;
    const float w1 = params.contains("w1") ? params["w1"].cast<float>() : 0.25f;

    problem.add_term<Op::EV>([w0] __device__(
                                 const auto& eh, const auto& iter, auto& x) {
        using ActiveT                    = ACTIVE_TYPE(eh);
        const Eigen::Vector3<ActiveT> x0 = x.template active<3>(eh, iter, 0);
        const Eigen::Vector3<ActiveT> x1 = x.template active<3>(eh, iter, 1);
        return w0 * (x0 - x1).squaredNorm();
    });

    problem.add_term<Op::EV>([w1] __device__(
                                 const auto& eh, const auto& iter, auto& x) {
        using ActiveT                    = ACTIVE_TYPE(eh);
        const Eigen::Vector3<ActiveT> x0 = x.template active<3>(eh, iter, 0);
        const Eigen::Vector3<ActiveT> x1 = x.template active<3>(eh, iter, 1);
        return w1 * (x0 - x1).squaredNorm();
    });
}

PYBIND11_MODULE(_rxmesh_diff_energy, m)
{
    pyrxmesh::require_compatible_runtime(m);

    m.def(
        "make_energy",
        [](py::object mesh, py::dict params) {
            return pyrxmesh::diff::make_scalar_energy<float, 3, VertexHandle>(
                mesh, std::move(params), add_edge_quadratic_terms);
        },
        py::arg("mesh"),
        py::arg("params") = py::dict());
}
