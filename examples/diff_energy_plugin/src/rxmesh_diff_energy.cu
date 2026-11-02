#include <pybind11/pybind11.h>

#include <utility>

#include "pyrxmesh/diff_plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;

template <typename Problem>
void add_edge_quadratic_terms(Problem& problem, py::dict params)
{
    const float w0 = params.contains("w0") ? params["w0"].cast<float>() : 0.5f;
    const float w1 = params.contains("w1") ? params["w1"].cast<float>() : 0.25f;

    problem.template add_term<Op::EV>([w0] __device__(const auto& eh,
                                                      const auto& iter,
                                                      auto&       x) {
        using ActiveT                    = ACTIVE_TYPE(eh);
        const Eigen::Vector3<ActiveT> x0 = x.template active<3>(eh, iter, 0);
        const Eigen::Vector3<ActiveT> x1 = x.template active<3>(eh, iter, 1);
        return w0 * (x0 - x1).squaredNorm();
    });

    problem.template add_term<Op::EV>([w1] __device__(const auto& eh,
                                                      const auto& iter,
                                                      auto&       x) {
        using ActiveT                    = ACTIVE_TYPE(eh);
        const Eigen::Vector3<ActiveT> x0 = x.template active<3>(eh, iter, 0);
        const Eigen::Vector3<ActiveT> x1 = x.template active<3>(eh, iter, 1);
        return w1 * (x0 - x1).squaredNorm();
    });
}

template <bool WithHessian, bool ProjectHessian = false>
std::shared_ptr<pyrxmesh::diff::ScalarEnergyBase> make_energy(py::object mesh,
                                                              py::dict   params)
{
    using Problem = pyrxmesh::diff::ScalarGradientProblem<float,
                                                          3,
                                                          VertexHandle,
                                                          WithHessian,
                                                          ProjectHessian>;
    return pyrxmesh::diff::
        make_scalar_energy<float, 3, VertexHandle, WithHessian, ProjectHessian>(
            std::move(mesh),
            std::move(params),
            add_edge_quadratic_terms<Problem>);
}

PYBIND11_MODULE(_rxmesh_diff_energy, m)
{
    pyrxmesh::require_compatible_runtime(m);

    m.def(
        "make_energy",
        [](py::object mesh,
           py::dict   params,
           bool       with_hessian,
           bool       projected_hessian) {
            if (projected_hessian && !with_hessian) {
                throw py::value_error(
                    "projected_hessian=True requires with_hessian=True.");
            }
            if (projected_hessian) {
                return make_energy<true, true>(std::move(mesh),
                                               std::move(params));
            }
            if (with_hessian) {
                return make_energy<true>(std::move(mesh), std::move(params));
            }
            return make_energy<false>(std::move(mesh), std::move(params));
        },
        py::arg("mesh"),
        py::arg("params") = py::dict(),
        py::kw_only(),
        py::arg("with_hessian")      = false,
        py::arg("projected_hessian") = false);
}
