#include "bindings/dispatch.h"
#include "bindings/py_attribute.h"
#include "bindings/py_dense_matrix.h"
#include "bindings/py_hessian_sparse_matrix.h"

#include "pyrxmesh/diff_plugin_api.h"

namespace pyrxmesh_py {

namespace {

void validate_energy_input(pyrxmesh::diff::ScalarEnergyBase&       energy,
                           const std::shared_ptr<PyAttributeBase>& opt_var)
{
    if (!opt_var) {
        throw std::invalid_argument(
            "ScalarEnergy.evaluate() requires an Attribute.");
    }
    if (opt_var->mesh_owner.get() != &energy.mesh()) {
        throw std::invalid_argument(
            "ScalarEnergy.evaluate() requires an Attribute from the same "
            "mesh.");
    }
    if (opt_var->dtype() != energy.dtype() ||
        opt_var->element_kind() != energy.element_kind() ||
        static_cast<int>(opt_var->dim()) != energy.variable_dim()) {
        throw std::invalid_argument(
            "ScalarEnergy.evaluate() Attribute type or shape does not match "
            "the energy.");
    }
    if (!opt_var->is_device_allocated()) {
        throw std::invalid_argument(
            "ScalarEnergy.evaluate() requires DEVICE storage.");
    }
}

template <typename T>
std::shared_ptr<PyDenseMatrix> gradient_matrix_view_typed(
    const std::shared_ptr<pyrxmesh::diff::ScalarEnergyBase>& energy)
{
    using MatT = rxmesh::DenseMatrix<T, Eigen::RowMajor>;

    auto matrix = std::make_shared<MatT>(
        MatT::device_view(energy->mesh(),
                          energy->element_count(),
                          energy->variable_dim(),
                          static_cast<T*>(energy->gradient_device_ptr())));
    auto output = std::make_shared<PyDenseMatrixT<T, Eigen::RowMajor>>(
        std::move(matrix), rxmesh::DEVICE);
    output->external_view  = true;
    output->external_owner = std::static_pointer_cast<void>(energy);
    output->read_only      = true;
    return output;
}

std::shared_ptr<PyDenseMatrix> gradient_matrix_view(
    const std::shared_ptr<pyrxmesh::diff::ScalarEnergyBase>& energy)
{
    if (!energy->has_evaluated()) {
        throw std::runtime_error(
            "ScalarEnergy.gradient_view is unavailable before evaluate().");
    }
    return dispatch_float_dtype(energy->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        return gradient_matrix_view_typed<T>(energy);
    });
}

void* dense_matrix_device_ptr(PyDenseMatrix& matrix)
{
    void* result = nullptr;
    with_typed_dense_matrix(matrix, [&](auto& typed) {
        result = static_cast<void*>(typed.matrix->data(rxmesh::DEVICE));
    });
    return result;
}

std::shared_ptr<PyDenseMatrix> gradient_snapshot(
    const std::shared_ptr<pyrxmesh::diff::ScalarEnergyBase>& energy,
    cudaStream_t                                             stream)
{
    auto output =
        make_dense_matrix_allow_empty_rows(energy->element_count(),
                                           energy->variable_dim(),
                                           dtype_name(energy->dtype()),
                                           static_cast<int>(rxmesh::DEVICE),
                                           "row_major");
    energy->gradient_snapshot(dense_matrix_device_ptr(*output), stream);
    return output;
}

std::shared_ptr<PySparseMatrix> hessian_snapshot(
    const std::shared_ptr<pyrxmesh::diff::ScalarEnergyBase>& energy,
    cudaStream_t                                             stream)
{
    // Validate availability before allocating an independent result.
    void* source = energy->hessian_matrix_ptr();
    auto  owner =
        energy->mesh_owner().cast<std::shared_ptr<rxmesh::RXMeshStatic>>();
    return dispatch_float_dtype(energy->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        return dispatch_int<1, 2, 3, 4, 6>(
            energy->variable_dim(),
            [&](auto k_tag) -> std::shared_ptr<PySparseMatrix> {
                constexpr int K = decltype(k_tag)::value;
                using HessT     = rxmesh::HessianSparseMatrix<T, K>;
                auto output     = std::make_shared<PyHessianSparseMatrix<T, K>>(
                    owner, 0, rxmesh::Op::VV);
                auto& hessian = *static_cast<HessT*>(source);
                // The supported fixed V/EV/FV stencils share this VV pattern.
                // copy_from copies values only. Keep the host copy fresh too.
                output->hessian->copy_from(
                    hessian, rxmesh::DEVICE, rxmesh::LOCATION_ALL, stream);
                using namespace rxmesh;
                CUDA_ERROR(cudaStreamSynchronize(stream));
                return output;
            });
    });
}

std::shared_ptr<PySparseMatrix> hessian_view(
    const std::shared_ptr<pyrxmesh::diff::ScalarEnergyBase>& energy)
{
    // Attribute evaluation is synchronous. No copy or extra synchronization
    // is needed here; the existing DLPack path exports the device pointers.
    void* source = energy->hessian_matrix_ptr();
    auto  owner =
        energy->mesh_owner().cast<std::shared_ptr<rxmesh::RXMeshStatic>>();
    return dispatch_float_dtype(energy->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        return dispatch_int<1, 2, 3, 4, 6>(
            energy->variable_dim(),
            [&](auto k_tag) -> std::shared_ptr<PySparseMatrix> {
                constexpr int K = decltype(k_tag)::value;
                using HessT     = rxmesh::HessianSparseMatrix<T, K>;
                auto borrowed =
                    std::shared_ptr<HessT>(energy, static_cast<HessT*>(source));
                return std::make_shared<PyHessianSparseMatrix<T, K>>(
                    owner, std::move(borrowed));
            });
    });
}

std::string energy_dtype_name(const pyrxmesh::diff::ScalarEnergyBase& energy)
{
    return dtype_name(energy.dtype());
}

std::string energy_element_kind_name(
    const pyrxmesh::diff::ScalarEnergyBase& energy)
{
    return element_kind_name(energy.element_kind());
}

}  // namespace

void register_diff_energy(py::module_& m)
{
    using Energy = pyrxmesh::diff::ScalarEnergyBase;

    py::class_<Energy, std::shared_ptr<Energy>>(m, "ScalarEnergy")
        .def_property_readonly("dtype", &energy_dtype_name)
        .def_property_readonly("element_kind", &energy_element_kind_name)
        .def_property_readonly("variable_dim", &Energy::variable_dim)
        .def_property_readonly("term_count", &Energy::term_count)
        .def_property_readonly("element_count", &Energy::element_count)
        .def_property_readonly("mesh", &Energy::mesh_owner)
        .def_property_readonly("has_evaluated", &Energy::has_evaluated)
        .def_property_readonly("has_gradient", &Energy::has_gradient)
        .def_property_readonly("has_hessian", &Energy::has_hessian)
        .def_property_readonly("hessian_enabled", &Energy::hessian_enabled)
        .def_property_readonly("projected_hessian", &Energy::projected_hessian)
        .def(
            "evaluate",
            [](const std::shared_ptr<Energy>&          self,
               const std::shared_ptr<PyAttributeBase>& opt_var,
               py::object                              stream,
               int                                     derivatives) {
                validate_energy_input(*self, opt_var);
                return self->evaluate(*opt_var->raw_attribute_base(),
                                      parse_cuda_stream_arg(std::move(stream)),
                                      derivatives);
            },
            py::arg("opt_var"),
            py::arg("stream") = py::none(),
            py::kw_only(),
            py::arg("derivatives") = 1,
            "Evaluate loss with derivatives=0 (value), 1 (gradient), or 2 "
            "(gradient and Hessian). Results use RXMesh linear element order.")
        .def(
            "evaluate_hessian",
            [](const std::shared_ptr<Energy>&          self,
               const std::shared_ptr<PyAttributeBase>& opt_var,
               py::object                              stream) {
                validate_energy_input(*self, opt_var);
                auto cuda_stream = parse_cuda_stream_arg(std::move(stream));
                self->evaluate(*opt_var->raw_attribute_base(), cuda_stream, 2);
                return hessian_snapshot(self, cuda_stream);
            },
            py::arg("opt_var"),
            py::arg("stream") = py::none(),
            "Evaluate loss, gradient, and Hessian; return an owned Hessian "
            "snapshot. DOF indices are k * linear_vertex + component.")
        .def(
            "_torch_forward",
            [](const std::shared_ptr<Energy>& self,
               std::uintptr_t                 input,
               std::uintptr_t                 gradient,
               std::uintptr_t                 term_losses,
               py::object                     stream) {
                self->evaluate_torch_buffers(
                    reinterpret_cast<void*>(input),
                    reinterpret_cast<void*>(gradient),
                    reinterpret_cast<void*>(term_losses),
                    parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("input_ptr"),
            py::arg("gradient_ptr"),
            py::arg("term_losses_ptr"),
            py::arg("stream"),
            "Internal Torch forward into PyTorch-owned CUDA buffers. A zero "
            "gradient pointer evaluates only the loss.")
        .def_property_readonly("loss", &Energy::loss_value)
        .def_property_readonly("gradient_view",
                               [](const std::shared_ptr<Energy>& self) {
                                   return gradient_matrix_view(self);
                               })
        .def_property_readonly(
            "hessian_view",
            &hessian_view,
            "Return a mutable zero-copy DEVICE view of the internal Hessian. "
            "The view retains the energy and is overwritten by subsequent "
            "second-order evaluations. Indices use RXMesh linear order.")
        .def(
            "gradient_snapshot",
            [](const std::shared_ptr<Energy>& self, py::object stream) {
                return gradient_snapshot(
                    self, parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("stream") = py::none(),
            "Return an owned row-major DEVICE copy of the latest gradient.")
        .def(
            "hessian_snapshot",
            [](const std::shared_ptr<Energy>& self, py::object stream) {
                return hessian_snapshot(
                    self, parse_cuda_stream_arg(std::move(stream)));
            },
            py::arg("stream") = py::none(),
            "Return an owned HOST+DEVICE Hessian copy after a second-order "
            "evaluation. Later evaluations do not change this snapshot.");
}

}  // namespace pyrxmesh_py
