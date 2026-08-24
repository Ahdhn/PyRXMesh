#pragma once

#include <memory>
#include <stdexcept>
#include <string>
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
                            cudaStream_t           stream) = 0;
    virtual double loss_value() const            = 0;
    virtual bool   has_evaluated() const         = 0;

    virtual void gradient_snapshot(void*        destination,
                                   cudaStream_t stream) const = 0;
    virtual void evaluate_torch_buffers(void*        input,
                                        void*        gradient_output,
                                        void*        term_loss_output,
                                        cudaStream_t stream)  = 0;

    virtual void* gradient_device_ptr() = 0;
};

/**
 * Gradient-only RXMesh scalar problem used by differentiable plugins.
 *
 * One instance may be called from different streams sequentially. Calls on
 * the same instance must not be made concurrently from native threads.
 */
template <typename T, int VariableDim, typename OptVarHandleT>
class ScalarGradientProblem final : public ScalarEnergyBase
{
   public:
    using ProblemT =
        rxmesh::DiffScalarProblem<T, VariableDim, OptVarHandleT, false>;
    using GradientMatrixT = rxmesh::DenseMatrix<T, Eigen::RowMajor>;

    explicit ScalarGradientProblem(
        py::handle  mesh_object,
        std::string opt_var_name_prefix = "pyrxmesh_diff_opt_var")
        : mesh_owner_(
              mesh_object.cast<std::shared_ptr<rxmesh::RXMeshStatic>>()),
          problem_(*mesh_owner_,
                   false,
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
        problem_.template add_term<op, false, block_threads>(
            std::forward<LambdaT>(term), oriented);
    }

    T evaluate(rxmesh::Attribute<T, OptVarHandleT>& opt_var,
               cudaStream_t                         stream = nullptr)
    {
        wait_for_previous_use(stream);
        ensure_internal_gradient();
        problem_.eval_terms_grad_only(&opt_var, stream);
        last_loss_     = problem_.get_current_loss(stream);
        has_evaluated_ = true;
        return last_loss_;
    }

    double evaluate(rxmesh::AttributeBase& opt_var,
                    cudaStream_t           stream) override
    {
        return static_cast<double>(
            evaluate(static_cast<rxmesh::Attribute<T, OptVarHandleT>&>(opt_var),
                     stream));
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
            auto gradient =
                GradientMatrixT::device_view(*mesh_owner_,
                                             element_count(),
                                             VariableDim,
                                             static_cast<T*>(gradient_output));
            problem_.eval_terms_grad_only(opt_var.get(), gradient, stream);
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
        require_evaluated("gradient_snapshot()");
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
        return static_cast<void*>(problem_.grad.data(rxmesh::DEVICE));
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
    cudaEvent_t                           last_use_event_ = nullptr;
};

template <typename T, int VariableDim, typename OptVarHandleT, typename Builder>
std::shared_ptr<ScalarEnergyBase> make_scalar_energy(py::object mesh,
                                                     py::dict   params,
                                                     Builder&&  builder)
{
    auto energy =
        std::make_shared<ScalarGradientProblem<T, VariableDim, OptVarHandleT>>(
            mesh);
    std::forward<Builder>(builder)(*energy, std::move(params));
    return energy;
}

}  // namespace pyrxmesh::diff
