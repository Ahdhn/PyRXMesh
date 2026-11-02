#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include "pyrxmesh/plugin_api.h"

#include "rxmesh/attribute.h"
#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/handle.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/rxmesh_static.h"

namespace pyrxmesh::diff {

namespace py = pybind11;

namespace detail {

inline rxmesh::DiffProblemMemoryOptions scalar_gradient_memory_options()
{
    rxmesh::DiffProblemMemoryOptions options;
    // Torch supplies its own gradient. Allocate the reusable low-level
    // gradient only if evaluate() is used.
    options.gradient_location     = rxmesh::LOCATION_NONE;
    options.opt_var_storage       = rxmesh::OptVarStorage::MetadataOnly;
    options.opt_var_location      = rxmesh::LOCATION_NONE;
    options.opt_var_layout        = rxmesh::SoA;
    options.term_loss_location    = rxmesh::DEVICE;
    options.unique_internal_names = true;
    return options;
}

}  // namespace detail

class ScalarEnergyBase
{
   public:
    virtual ~ScalarEnergyBase() = default;

    virtual DType       dtype() const         = 0;
    virtual ElementKind element_kind() const  = 0;
    virtual int         variable_dim() const  = 0;
    virtual int         element_count() const = 0;
    virtual size_t      term_count() const    = 0;

    virtual rxmesh::RXMeshStatic& mesh()             = 0;
    virtual py::object            mesh_owner() const = 0;

    virtual double evaluate(rxmesh::AttributeBase& opt_var,
                            cudaStream_t           stream,
                            int                    derivatives = 1) = 0;
    virtual double loss_value() const            = 0;
    virtual bool   has_evaluated() const         = 0;
    virtual bool   has_gradient() const          = 0;
    virtual bool   has_hessian() const           = 0;
    virtual bool   hessian_enabled() const       = 0;
    virtual bool   projected_hessian() const     = 0;

    virtual void gradient_snapshot(void*        destination,
                                   cudaStream_t stream) const = 0;
    virtual void evaluate_torch_buffers(void*        input,
                                        void*        gradient_output,
                                        void*        term_loss_output,
                                        cudaStream_t stream)  = 0;

    virtual void* gradient_device_ptr() = 0;
    // Borrowed native matrix for an immediate, owned snapshot in the binding.
    virtual void* hessian_matrix_ptr() = 0;
};

/**
 * RXMesh scalar problem with optional second-order differentiation.
 *
 * One instance may be called from different streams sequentially. Calls on
 * the same instance must not be made concurrently from native threads.
 */
template <typename T,
          int VariableDim,
          typename OptVarHandleT,
          bool WithHessian    = false,
          bool ProjectHessian = false>
class ScalarGradientProblem final : public ScalarEnergyBase
{
    static_assert(!ProjectHessian || WithHessian,
                  "Projected Hessians require WithHessian=true.");
    static_assert(!WithHessian ||
                      std::is_same_v<OptVarHandleT, rxmesh::VertexHandle>,
                  "Assembled Hessians currently require vertex variables.");
    static_assert(!WithHessian || VariableDim == 1 || VariableDim == 2 ||
                      VariableDim == 3 || VariableDim == 4 || VariableDim == 6,
                  "Hessians support variable dimensions 1, 2, 3, 4, or 6.");

   public:
    using ProblemT =
        rxmesh::DiffScalarProblem<T, VariableDim, OptVarHandleT, WithHessian>;
    using GradientMatrixT = rxmesh::DenseMatrix<T, Eigen::RowMajor>;

    explicit ScalarGradientProblem(
        py::handle  mesh_object,
        std::string opt_var_name_prefix = "pyrxmesh_diff_opt_var")
        : mesh_owner_(
              mesh_object.cast<std::shared_ptr<rxmesh::RXMeshStatic>>()),
          problem_(*mesh_owner_,
                   WithHessian,
                   detail::scalar_gradient_memory_options(),
                   opt_var_name_prefix.empty() ? "pyrxmesh_diff_opt_var" :
                                                 std::move(opt_var_name_prefix))
    {
    }

    ~ScalarGradientProblem() override
    {
        if (last_use_event_) {
            (void)cudaEventSynchronize(last_use_event_);
            (void)cudaEventDestroy(last_use_event_);
        }
    }

    /**
     * Register a read-only scalar energy lambda.
     */
    template <rxmesh::Op op,
              uint32_t   block_threads = 256,
              typename LambdaT         = void>
    void add_term(LambdaT&& term, bool oriented = false)
    {
        static_assert(!WithHessian || op == rxmesh::Op::V ||
                          op == rxmesh::Op::EV || op == rxmesh::Op::FV,
                      "Assembled Hessians currently support V, EV, and FV "
                      "terms with fixed vertex stencils.");
        problem_.template add_term<op, ProjectHessian, block_threads>(
            std::forward<LambdaT>(term), oriented);
    }

    T evaluate(rxmesh::Attribute<T, OptVarHandleT>& opt_var,
               cudaStream_t                         stream      = nullptr,
               int                                  derivatives = 1)
    {
        if (derivatives < 0 || derivatives > 2) {
            throw std::invalid_argument("derivatives must be 0, 1, or 2.");
        }
        if (derivatives == 2 && !WithHessian) {
            throw std::invalid_argument(
                "This energy was built without Hessians. Create it with "
                "with_hessian=True.");
        }
        wait_for_previous_use(stream);
        has_evaluated_ = false;
        has_gradient_  = false;
        has_hessian_   = false;
        if (derivatives == 0) {
            problem_.eval_terms_passive(&opt_var, stream);
        } else {
            ensure_internal_gradient();
            if (derivatives == 1) {
                problem_.eval_terms_grad_only(&opt_var, stream);
            } else if constexpr (WithHessian) {
                // eval_terms() requires an owned opt_var. Our problem uses
                // MetadataOnly storage, so pass the external Attribute to
                // each term just as the gradient-only path does.
                problem_.grad.reset(T(0), rxmesh::DEVICE, stream);
                problem_.hess->reset(T(0), rxmesh::DEVICE, stream);
                for (auto& term : problem_.terms) {
                    term->eval_active(
                        opt_var, problem_.face_interact_vertex.get(), stream);
                }
            }
        }
        // Also covers an empty term list. The low-level API is synchronous.
        record_use(stream);
        last_loss_ = problem_.get_current_loss(stream);
        {
            using namespace rxmesh;
            CUDA_ERROR(cudaStreamSynchronize(stream));
        }
        has_evaluated_ = true;
        has_gradient_  = derivatives >= 1;
        has_hessian_   = derivatives == 2;
        return last_loss_;
    }

    double evaluate(rxmesh::AttributeBase& opt_var,
                    cudaStream_t           stream,
                    int                    derivatives = 1) override
    {
        return static_cast<double>(
            evaluate(static_cast<rxmesh::Attribute<T, OptVarHandleT>&>(opt_var),
                     stream,
                     derivatives));
    }

    void evaluate_torch_buffers(void*        input,
                                void*        gradient_output,
                                void*        term_loss_output,
                                cudaStream_t stream) override
    {
        using namespace rxmesh;

        wait_for_previous_use(stream);
        const size_t input_bytes =
            static_cast<size_t>(element_count()) * VariableDim * sizeof(T);
        auto opt_var = problem_.opt_var;
        if (!opt_var->attach_device_buffer(static_cast<T*>(input),
                                           input_bytes)) {
            throw std::invalid_argument(
                "Could not attach the Torch input buffer.");
        }

        try {
            if (gradient_output == nullptr) {
                problem_.eval_terms_passive(opt_var.get(), stream);
            } else {
                auto gradient = GradientMatrixT::device_view(
                    *mesh_owner_,
                    element_count(),
                    VariableDim,
                    static_cast<T*>(gradient_output));
                problem_.eval_terms_grad_only(opt_var.get(), gradient, stream);
            }
            problem_.get_current_loss_device(
                static_cast<T*>(term_loss_output), term_count(), stream);
            CUDA_ERROR(cudaGetLastError());
            opt_var->detach_device_buffer();
            record_use(stream);
        } catch (...) {
            opt_var->detach_device_buffer();
            throw;
        }
    }

    void gradient_snapshot(void*        destination,
                           cudaStream_t stream) const override
    {
        require_gradient("gradient_snapshot()");
        const size_t count = static_cast<size_t>(element_count()) * VariableDim;
        if (count != 0) {
            using namespace rxmesh;
            CUDA_ERROR(cudaMemcpyAsync(destination,
                                       problem_.grad.data(DEVICE),
                                       count * sizeof(T),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
            CUDA_ERROR(cudaStreamSynchronize(stream));
        }
    }

    T loss() const
    {
        require_evaluated("loss");
        return last_loss_;
    }

    double loss_value() const override
    {
        return static_cast<double>(loss());
    }

    bool has_evaluated() const override
    {
        return has_evaluated_;
    }

    bool has_gradient() const override
    {
        return has_gradient_;
    }
    bool has_hessian() const override
    {
        return has_hessian_;
    }
    bool hessian_enabled() const override
    {
        return WithHessian;
    }
    bool projected_hessian() const override
    {
        return ProjectHessian;
    }

    DType dtype() const override
    {
        return dtype_of<T>::value;
    }

    ElementKind element_kind() const override
    {
        return element_kind_of<OptVarHandleT>::value;
    }

    int variable_dim() const override
    {
        return VariableDim;
    }

    int element_count() const override
    {
        return static_cast<int>(mesh_owner_->get_num_elements<OptVarHandleT>());
    }

    size_t term_count() const override
    {
        return problem_.get_num_terms();
    }

    rxmesh::RXMeshStatic& mesh() override
    {
        return *mesh_owner_;
    }

    py::object mesh_owner() const override
    {
        return py::cast(mesh_owner_);
    }

    void* gradient_device_ptr() override
    {
        require_gradient("gradient_view");
        return static_cast<void*>(problem_.grad.data(rxmesh::DEVICE));
    }

    void* hessian_matrix_ptr() override
    {
        if (!has_hessian_) {
            throw std::runtime_error(
                "ScalarEnergy.hessian_snapshot() requires evaluate(..., "
                "derivatives=2) or evaluate_hessian().");
        }
        return static_cast<void*>(problem_.hess.get());
    }

   private:
    void ensure_internal_gradient()
    {
        if ((problem_.grad.get_allocated() & rxmesh::DEVICE) == 0) {
            problem_.grad = GradientMatrixT(
                *mesh_owner_, element_count(), VariableDim, rxmesh::DEVICE);
        }
    }

    void require_evaluated(const char* name) const
    {
        if (!has_evaluated_) {
            throw std::runtime_error(std::string("ScalarEnergy.") + name +
                                     " is unavailable before evaluate().");
        }
    }

    void require_gradient(const char* name) const
    {
        require_evaluated(name);
        if (!has_gradient_) {
            throw std::runtime_error(
                std::string("ScalarEnergy.") + name +
                " requires evaluate() with derivatives=1 or 2.");
        }
    }

    void wait_for_previous_use(cudaStream_t stream)
    {
        if (last_use_event_) {
            using namespace rxmesh;
            CUDA_ERROR(cudaStreamWaitEvent(stream, last_use_event_, 0));
        }
    }

    void record_use(cudaStream_t stream)
    {
        using namespace rxmesh;
        if (!last_use_event_) {
            CUDA_ERROR(cudaEventCreateWithFlags(&last_use_event_,
                                                cudaEventDisableTiming));
        }
        CUDA_ERROR(cudaEventRecord(last_use_event_, stream));
    }

    std::shared_ptr<rxmesh::RXMeshStatic> mesh_owner_;
    ProblemT                              problem_;
    T                                     last_loss_      = T(0);
    bool                                  has_evaluated_  = false;
    bool                                  has_gradient_   = false;
    bool                                  has_hessian_    = false;
    cudaEvent_t                           last_use_event_ = nullptr;
};

template <typename T,
          int VariableDim,
          typename OptVarHandleT,
          bool WithHessian    = false,
          bool ProjectHessian = false,
          typename Builder>
std::shared_ptr<ScalarEnergyBase> make_scalar_energy(py::object mesh,
                                                     py::dict   params,
                                                     Builder&&  builder)
{
    auto energy = std::make_shared<ScalarGradientProblem<T,
                                                         VariableDim,
                                                         OptVarHandleT,
                                                         WithHessian,
                                                         ProjectHessian>>(mesh);
    std::forward<Builder>(builder)(*energy, std::move(params));
    return energy;
}

}  // namespace pyrxmesh::diff
