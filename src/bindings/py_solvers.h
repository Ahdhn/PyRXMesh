#pragma once

#include "bindings/py_dense_matrix.h"
#include "bindings/py_sparse_matrix.h"

#include "rxmesh/matrix/cg_solver.h"
#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/cudss_cholesky_solver.h"
#include "rxmesh/matrix/lu_solver.h"
#include "rxmesh/matrix/pcg_solver.h"
#include "rxmesh/matrix/qr_solver.h"

namespace pyrxmesh_py {

enum class DirectSolverKind
{
    Cholesky,
    QR,
    LU,
    cuDSSCholesky,
};

namespace detail {

inline void validate_solver_rhs(const PySparseMatrix& matrix,
                                int                   unknown_dim,
                                const PyDenseMatrix&  rhs,
                                const char*           api)
{
    if (matrix.dtype() != rhs.dtype()) {
        throw std::invalid_argument(std::string(api) +
                                    " rhs dtype must match the SparseMatrix "
                                    "dtype.");
    }
    if (rhs.rows() != matrix.rows()) {
        throw std::invalid_argument(std::string(api) +
                                    " rhs rows must match SparseMatrix rows.");
    }
    if (unknown_dim > 0 && rhs.cols() != unknown_dim) {
        throw std::invalid_argument(
            std::string(api) + " rhs columns must match solver unknown_dim.");
    }
}

inline void validate_solver_solution(const PySparseMatrix& matrix,
                                     const PyDenseMatrix&  solution,
                                     const PyDenseMatrix&  rhs,
                                     const char*           api)
{
    if (solution.dtype() != rhs.dtype()) {
        throw std::invalid_argument(std::string(api) +
                                    " solution dtype must match rhs dtype.");
    }
    if (solution.rows() != matrix.cols() || solution.cols() != rhs.cols()) {
        throw std::invalid_argument(std::string(api) +
                                    " solution shape must be "
                                    "(SparseMatrix.cols, rhs.cols).");
    }
}

inline void validate_solver_system(const PySparseMatrix& matrix,
                                   int                   unknown_dim,
                                   const PyDenseMatrix&  rhs,
                                   const PyDenseMatrix&  solution,
                                   const char*           api)
{
    validate_solver_rhs(matrix, unknown_dim, rhs, api);
    validate_solver_solution(matrix, solution, rhs, api);
}

template <typename SolveIntoFn>
inline std::shared_ptr<PyDenseMatrix> make_solution_and_solve(
    const PySparseMatrix& matrix,
    int                   unknown_dim,
    PyDenseMatrix&        rhs,
    py::object            initial_guess,
    bool                  run_pre_solve,
    const char*           api,
    SolveIntoFn&&         solve_into_fn)
{
    validate_solver_rhs(matrix, unknown_dim, rhs, api);
    auto solution = make_dense_matrix(matrix.cols(),
                                      rhs.cols(),
                                      rhs.dtype(),
                                      static_cast<int>(rxmesh::LOCATION_ALL),
                                      "col_major");
    if (initial_guess.is_none()) {
        solution->reset(
            py::float_(0.0), static_cast<int>(rxmesh::LOCATION_ALL), nullptr);
    } else {
        auto initial = initial_guess.cast<std::shared_ptr<PyDenseMatrix>>();
        validate_solver_solution(matrix, *initial, rhs, api);
        solution->copy_from(*initial,
                            static_cast<int>(rxmesh::LOCATION_ALL),
                            static_cast<int>(rxmesh::LOCATION_ALL),
                            nullptr);
    }
    solve_into_fn(rhs, *solution, run_pre_solve);
    return solution;
}

}  // namespace detail


// -----------------------------------------------------------------------------
// Iterative solvers (CG, PCG)
// -----------------------------------------------------------------------------

template <template <typename, int> class SolverT>
struct PyIterativeSolverBase
    : std::enable_shared_from_this<PyIterativeSolverBase<SolverT>>
{
    virtual ~PyIterativeSolverBase() = default;

    virtual std::string name() const                                    = 0;
    virtual int         iter_taken() const                              = 0;
    virtual py::object  start_residual() const                          = 0;
    virtual py::object  final_residual() const                          = 0;
    virtual void pre_solve(PyDenseMatrix& rhs, PyDenseMatrix& solution) = 0;
    virtual void solve_into(PyDenseMatrix& rhs,
                            PyDenseMatrix& solution,
                            bool           run_pre_solve)                         = 0;

    std::shared_ptr<PyDenseMatrix> solve(PyDenseMatrix& rhs,
                                         py::object     initial_guess,
                                         bool           run_pre_solve)
    {
        return detail::make_solution_and_solve(
            *matrix,
            unknown_dim,
            rhs,
            std::move(initial_guess),
            run_pre_solve,
            "Iterative solver",
            [this](PyDenseMatrix& r, PyDenseMatrix& s, bool ps) {
                this->solve_into(r, s, ps);
            });
    }

    std::shared_ptr<PySparseMatrix> matrix;
    int                             unknown_dim = 1;
};

template <typename T, template <typename, int> class SolverT>
struct PyIterativeSolverT final : PyIterativeSolverBase<SolverT>
{
    using SolverImplT = SolverT<T, Eigen::ColMajor>;
    std::shared_ptr<SolverImplT> solver;

    std::string name() const override
    {
        return solver->name();
    }
    int iter_taken() const override
    {
        return solver->iter_taken();
    }
    py::object start_residual() const override
    {
        return py::cast(solver->start_residual());
    }
    py::object final_residual() const override
    {
        return py::cast(solver->final_residual());
    }

    void pre_solve(PyDenseMatrix& rhs, PyDenseMatrix& solution) override
    {
        using namespace rxmesh;
        detail::validate_solver_system(*this->matrix,
                                       this->unknown_dim,
                                       rhs,
                                       solution,
                                       "Iterative solver");

        auto& rhs_t      = as_typed_dense<T>(rhs, "Iterative solver");
        auto& solution_t = as_typed_dense<T>(solution, "Iterative solver");
        solver->pre_solve(*rhs_t.matrix, *solution_t.matrix);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
    }

    void solve_into(PyDenseMatrix& rhs,
                    PyDenseMatrix& solution,
                    bool           run_pre_solve) override
    {
        using namespace rxmesh;
        detail::validate_solver_system(*this->matrix,
                                       this->unknown_dim,
                                       rhs,
                                       solution,
                                       "Iterative solver");

        if (run_pre_solve) {
            pre_solve(rhs, solution);
        }

        auto& rhs_t      = as_typed_dense<T>(rhs, "Iterative solver");
        auto& solution_t = as_typed_dense<T>(solution, "Iterative solver");
        solver->solve(*rhs_t.matrix, *solution_t.matrix);

        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        solution.move(rxmesh::DEVICE, rxmesh::HOST, nullptr);
    }
};

using PyCGSolver  = PyIterativeSolverBase<rxmesh::CGSolver>;
using PyPCGSolver = PyIterativeSolverBase<rxmesh::PCGSolver>;

template <template <typename, int> class SolverT>
inline std::shared_ptr<PyIterativeSolverBase<SolverT>> make_iterative_solver(
    std::shared_ptr<PySparseMatrix> matrix,
    int                             unknown_dim,
    int                             max_iter,
    py::object                      abs_tol,
    py::object                      rel_tol,
    int                             reset_residual_freq)
{
    if (!matrix) {
        throw std::invalid_argument("Solver requires a SparseMatrix.");
    }
    if (matrix->rows() != matrix->cols()) {
        throw std::invalid_argument(
            "Iterative solvers require a square SparseMatrix.");
    }
    if (unknown_dim <= 0) {
        throw std::invalid_argument("Solver unknown_dim must be positive.");
    }
    matrix->ensure_device_readable();

    std::shared_ptr<PyIterativeSolverBase<SolverT>> result;
    with_float_double_sparse_matrix(
        *matrix, "Iterative solvers", [&](auto& sparse_mat) {
            using SparseMatT = std::decay_t<decltype(sparse_mat)>;
            using T          = typename SparseMatT::element_type::Type;
            auto wrapper = std::make_shared<PyIterativeSolverT<T, SolverT>>();
            wrapper->matrix      = matrix;
            wrapper->unknown_dim = unknown_dim;
            wrapper->solver = std::make_shared<SolverT<T, Eigen::ColMajor>>(
                *sparse_mat,
                unknown_dim,
                max_iter,
                abs_tol.cast<T>(),
                rel_tol.cast<T>(),
                reset_residual_freq);
            result = std::static_pointer_cast<PyIterativeSolverBase<SolverT>>(
                wrapper);
        });
    return result;
}


// -----------------------------------------------------------------------------
// Direct solvers (Cholesky, QR, LU)
// -----------------------------------------------------------------------------

template <template <typename, int> class SolverT, DirectSolverKind Kind>
struct PyDirectSolverBase
    : std::enable_shared_from_this<PyDirectSolverBase<SolverT, Kind>>
{
    virtual ~PyDirectSolverBase() = default;

    virtual std::string name() const                          = 0;
    virtual void        pre_solve(rxmesh::RXMeshStatic& mesh) = 0;
    virtual void        solve_into(PyDenseMatrix& rhs,
                                   PyDenseMatrix& solution,
                                   bool           run_pre_solve)        = 0;

    std::string permute() const
    {
        return rxmesh::permute_method_to_string(permute_method);
    }

    bool is_factorized() const
    {
        return factorized;
    }

    std::shared_ptr<PyDenseMatrix> solve(PyDenseMatrix& rhs,
                                         py::object     initial_guess,
                                         bool           run_pre_solve)
    {
        return detail::make_solution_and_solve(
            *matrix,
            0,
            rhs,
            std::move(initial_guess),
            run_pre_solve,
            "Direct solver",
            [this](PyDenseMatrix& r, PyDenseMatrix& s, bool ps) {
                this->solve_into(r, s, ps);
            });
    }

    std::shared_ptr<PySparseMatrix> matrix;
    rxmesh::PermuteMethod permute_method = rxmesh::PermuteMethod::NONE;
    bool                  factorized     = false;
};

template <typename T,
          template <typename, int> class SolverT,
          DirectSolverKind Kind>
struct PyDirectSolverT final : PyDirectSolverBase<SolverT, Kind>
{
    using SolverImplT = SolverT<rxmesh::SparseMatrix<T>, Eigen::ColMajor>;
    std::shared_ptr<SolverImplT> solver;

    std::string name() const override
    {
        return solver->name();
    }

    void pre_solve(rxmesh::RXMeshStatic& mesh) override
    {
        using namespace rxmesh;
        prepare_matrix_for_device_solver();
        solver->pre_solve(mesh);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        this->factorized = true;
    }

    void solve_into(PyDenseMatrix& rhs,
                    PyDenseMatrix& solution,
                    bool           run_pre_solve) override
    {
        using namespace rxmesh;
        detail::validate_solver_system(
            *this->matrix, 0, rhs, solution, "Direct solver");

        if constexpr (Kind == DirectSolverKind::LU) {
            this->matrix->ensure_host_readable();
        } else {
            prepare_matrix_for_device_solver();
        }

        if (run_pre_solve && !this->factorized && this->matrix->mesh_owner) {
            pre_solve(*this->matrix->mesh_owner);
        }

        auto& rhs_t      = as_typed_dense<T>(rhs, "Direct solver");
        auto& solution_t = as_typed_dense<T>(solution, "Direct solver");

        if constexpr (Kind == DirectSolverKind::Cholesky ||
                      Kind == DirectSolverKind::QR) {
            if (this->factorized) {
                solver->solve(*rhs_t.matrix, *solution_t.matrix);
            } else {
                solver->solve_hl_api(*rhs_t.matrix, *solution_t.matrix);
            }
        } else {
            solver->solve(*rhs_t.matrix, *solution_t.matrix);
        }

        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        if constexpr (Kind != DirectSolverKind::LU) {
            solution.move(rxmesh::DEVICE, rxmesh::HOST, nullptr);
        }
    }

   private:
    void prepare_matrix_for_device_solver()
    {
        this->matrix->ensure_host_readable();
        this->matrix->ensure_device_readable();
    }
};

using PyCholeskySolver =
    PyDirectSolverBase<rxmesh::CholeskySolver, DirectSolverKind::Cholesky>;
using PyQRSolver = PyDirectSolverBase<rxmesh::QRSolver, DirectSolverKind::QR>;
using PyLUSolver = PyDirectSolverBase<rxmesh::LUSolver, DirectSolverKind::LU>;

template <template <typename, int> class SolverT, DirectSolverKind Kind>
inline std::shared_ptr<PyDirectSolverBase<SolverT, Kind>> make_direct_solver(
    std::shared_ptr<PySparseMatrix> matrix,
    std::string                     permute)
{
    if (!matrix) {
        throw std::invalid_argument("Direct solver requires a SparseMatrix.");
    }
    if (matrix->rows() != matrix->cols()) {
        throw std::invalid_argument(
            "Direct solvers require a square SparseMatrix.");
    }

    const rxmesh::PermuteMethod permute_method =
        rxmesh::string_to_permute_method(std::move(permute));

    std::shared_ptr<PyDirectSolverBase<SolverT, Kind>> result;
    with_float_double_sparse_matrix(
        *matrix, "Direct solvers", [&](auto& sparse_mat) {
            using SparseMatT = std::decay_t<decltype(sparse_mat)>;
            using T          = typename SparseMatT::element_type::Type;
            auto wrapper =
                std::make_shared<PyDirectSolverT<T, SolverT, Kind>>();
            wrapper->matrix         = matrix;
            wrapper->permute_method = permute_method;
            wrapper->solver         = std::make_shared<
                        SolverT<rxmesh::SparseMatrix<T>, Eigen::ColMajor>>(
                sparse_mat.get(), permute_method);
            result =
                std::static_pointer_cast<PyDirectSolverBase<SolverT, Kind>>(
                    wrapper);
        });
    return result;
}


// -----------------------------------------------------------------------------
// cuDSS direct solver (optional, requires PYRXMESH_USE_CUDSS=ON)
// -----------------------------------------------------------------------------

#ifdef USE_CUDSS

template <template <typename, int> class SolverT, DirectSolverKind Kind>
struct PyCuDSSSolverBase
    : std::enable_shared_from_this<PyCuDSSSolverBase<SolverT, Kind>>
{
    virtual ~PyCuDSSSolverBase() = default;

    virtual std::string name() const                       = 0;
    virtual void        pre_solve(rxmesh::RXMeshStatic& mesh,
                                  PyDenseMatrix&        rhs,
                                  PyDenseMatrix&        solution) = 0;
    virtual void        solve_into(PyDenseMatrix& rhs,
                                   PyDenseMatrix& solution,
                                   bool           run_pre_solve)     = 0;

    std::string permute() const
    {
        return rxmesh::permute_method_to_string(permute_method);
    }

    bool is_factorized() const
    {
        return factorized;
    }

    std::shared_ptr<PyDenseMatrix> solve(PyDenseMatrix& rhs,
                                         py::object     initial_guess,
                                         bool           run_pre_solve)
    {
        return detail::make_solution_and_solve(
            *matrix,
            0,
            rhs,
            std::move(initial_guess),
            run_pre_solve,
            "cuDSS direct solver",
            [this](PyDenseMatrix& r, PyDenseMatrix& s, bool ps) {
                this->solve_into(r, s, ps);
            });
    }

    std::shared_ptr<PySparseMatrix> matrix;
    rxmesh::PermuteMethod permute_method = rxmesh::PermuteMethod::NONE;
    bool                  factorized     = false;
};

template <typename T,
          template <typename, int> class SolverT,
          DirectSolverKind Kind>
struct PyCuDSSSolverT final : PyCuDSSSolverBase<SolverT, Kind>
{
    using SolverImplT = SolverT<rxmesh::SparseMatrix<T>, Eigen::ColMajor>;
    std::shared_ptr<SolverImplT> solver;

    std::string name() const override
    {
        return solver->name();
    }

    void pre_solve(rxmesh::RXMeshStatic& mesh,
                   PyDenseMatrix&        rhs,
                   PyDenseMatrix&        solution) override
    {
        detail::validate_solver_system(
            *this->matrix, 0, rhs, solution, "cuDSS direct solver");
        this->matrix->ensure_host_readable();
        this->matrix->ensure_device_readable();

        auto& rhs_t      = as_typed_dense<T>(rhs, "cuDSS direct solver");
        auto& solution_t = as_typed_dense<T>(solution, "cuDSS direct solver");
        solver->pre_solve(mesh, *rhs_t.matrix, *solution_t.matrix);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        this->factorized = true;
    }

    void solve_into(PyDenseMatrix& rhs,
                    PyDenseMatrix& solution,
                    bool           run_pre_solve) override
    {
        detail::validate_solver_system(
            *this->matrix, 0, rhs, solution, "cuDSS direct solver");
        if (run_pre_solve && !this->factorized) {
            if (!this->matrix->mesh_owner) {
                throw std::invalid_argument(
                    "cuDSSCholeskySolver.solve_into(pre_solve=True) needs a "
                    "mesh-owned SparseMatrix or an explicit pre_solve(mesh, "
                    "rhs, solution) call.");
            }
            pre_solve(*this->matrix->mesh_owner, rhs, solution);
        }
        if (!this->factorized) {
            throw std::runtime_error(
                "cuDSSCholeskySolver.solve_into() requires pre_solve() before "
                "solve.");
        }

        auto& rhs_t      = as_typed_dense<T>(rhs, "cuDSS direct solver");
        auto& solution_t = as_typed_dense<T>(solution, "cuDSS direct solver");
        solver->solve(*rhs_t.matrix, *solution_t.matrix);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));

        solution.move(rxmesh::DEVICE, rxmesh::HOST, nullptr);
    }
};

using PycuDSSCholeskySolver =
    PyCuDSSSolverBase<rxmesh::cuDSSCholeskySolver,
                      DirectSolverKind::cuDSSCholesky>;

template <template <typename, int> class SolverT, DirectSolverKind Kind>
inline std::shared_ptr<PyCuDSSSolverBase<SolverT, Kind>> make_cudss_solver(
    std::shared_ptr<PySparseMatrix> matrix,
    std::string                     permute)
{
    if (!matrix) {
        throw std::invalid_argument(
            "cuDSS direct solver requires a SparseMatrix.");
    }
    if (matrix->rows() != matrix->cols()) {
        throw std::invalid_argument(
            "cuDSS direct solvers require a square SparseMatrix.");
    }
    matrix->ensure_host_readable();
    matrix->ensure_device_readable();

    const rxmesh::PermuteMethod permute_method =
        rxmesh::string_to_permute_method(std::move(permute));

    std::shared_ptr<PyCuDSSSolverBase<SolverT, Kind>> result;
    with_float_double_sparse_matrix(
        *matrix, "cuDSS direct solvers", [&](auto& sparse_mat) {
            using SparseMatT = std::decay_t<decltype(sparse_mat)>;
            using T          = typename SparseMatT::element_type::Type;
            auto wrapper = std::make_shared<PyCuDSSSolverT<T, SolverT, Kind>>();
            wrapper->matrix         = matrix;
            wrapper->permute_method = permute_method;
            wrapper->solver         = std::make_shared<
                        SolverT<rxmesh::SparseMatrix<T>, Eigen::ColMajor>>(
                sparse_mat.get(), permute_method);
            result = std::static_pointer_cast<PyCuDSSSolverBase<SolverT, Kind>>(
                wrapper);
        });
    return result;
}

#endif  // USE_CUDSS

struct PyUnavailableCuDSSCholeskySolver
{
    PyUnavailableCuDSSCholeskySolver(std::shared_ptr<PySparseMatrix>,
                                     std::string)
    {
        throw std::runtime_error(
            "cuDSSCholeskySolver is available only when PyRXMesh is built with "
            "PYRXMESH_USE_CUDSS=ON.");
    }
};

}  // namespace pyrxmesh_py
