#pragma once

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

template <template <typename, int> class SolverT>
using IterativeSolverVariant =
    std::variant<std::shared_ptr<SolverT<float, Eigen::ColMajor>>,
                 std::shared_ptr<SolverT<double, Eigen::ColMajor>>>;

template <template <typename, int> class SolverT>
using DirectSolverVariant = std::variant<
    std::shared_ptr<SolverT<rxmesh::SparseMatrix<float>, Eigen::ColMajor>>,
    std::shared_ptr<SolverT<rxmesh::SparseMatrix<double>, Eigen::ColMajor>>>;

template <typename T>
using DenseMatrixPtr = std::shared_ptr<rxmesh::DenseMatrix<T, Eigen::ColMajor>>;

template <template <typename, int> class SolverT>
struct PyIterativeSolver
{
    PyIterativeSolver(std::shared_ptr<PySparseMatrix> matrix_in,
                      int                             unknown_dim,
                      int                             max_iter,
                      py::object                      abs_tol,
                      py::object                      rel_tol,
                      int                             reset_residual_freq)
        : matrix(std::move(matrix_in)), unknown_dim(unknown_dim)
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

        with_float_double_sparse_matrix(
            *matrix,
            "Iterative solvers",
            [&](auto& sparse_mat) {
                using SparseMatT = std::decay_t<decltype(sparse_mat)>;
                using T          = typename SparseMatT::element_type::Type;
                solver = std::make_shared<SolverT<T, Eigen::ColMajor>>(
                    *sparse_mat,
                    unknown_dim,
                    max_iter,
                    abs_tol.cast<T>(),
                    rel_tol.cast<T>(),
                    reset_residual_freq);
            });
    }

    std::string name()
    {
        return std::visit([](auto& solver_ptr) { return solver_ptr->name(); },
                          solver);
    }

    int iter_taken()
    {
        return std::visit(
            [](auto& solver_ptr) { return solver_ptr->iter_taken(); }, solver);
    }

    py::object start_residual()
    {
        return std::visit(
            [](auto& solver_ptr) -> py::object {
                return py::cast(solver_ptr->start_residual());
            },
            solver);
    }

    py::object final_residual()
    {
        return std::visit(
            [](auto& solver_ptr) -> py::object {
                return py::cast(solver_ptr->final_residual());
            },
            solver);
    }

    void pre_solve(PyDenseMatrix& rhs, PyDenseMatrix& solution)
    {
        using namespace rxmesh;
        validate_system(rhs, solution);

        std::visit(
            [&](auto& solver_ptr) {
                pre_solve_typed(solver_ptr, rhs, solution);
            },
            solver);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
    }

    void solve_into(PyDenseMatrix& rhs,
                    PyDenseMatrix& solution,
                    bool           run_pre_solve)
    {
        using namespace rxmesh;
        validate_system(rhs, solution);

        if (run_pre_solve) {
            pre_solve(rhs, solution);
        }

        std::visit(
            [&](auto& solver_ptr) { solve_typed(solver_ptr, rhs, solution); },
            solver);

        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        solution.move(rxmesh::DEVICE, rxmesh::HOST);
    }

    std::shared_ptr<PyDenseMatrix> solve(PyDenseMatrix& rhs,
                                         py::object     initial_guess,
                                         bool           run_pre_solve)
    {
        validate_rhs(rhs);
        auto solution = std::make_shared<PyDenseMatrix>(
            rhs.dtype(),
            matrix->cols(),
            rhs.cols(),
            static_cast<int>(rxmesh::LOCATION_ALL),
            "col_major");

        if (initial_guess.is_none()) {
            solution->reset(py::float_(0.0),
                            static_cast<int>(rxmesh::LOCATION_ALL));
        } else {
            auto initial = initial_guess.cast<std::shared_ptr<PyDenseMatrix>>();
            validate_solution(*initial, rhs);
            solution->copy_from(*initial,
                                static_cast<int>(rxmesh::LOCATION_ALL),
                                static_cast<int>(rxmesh::LOCATION_ALL));
        }

        solve_into(rhs, *solution, run_pre_solve);
        return solution;
    }

    std::shared_ptr<PySparseMatrix> matrix;
    int                             unknown_dim;
    IterativeSolverVariant<SolverT> solver;

   private:
    template <typename SolverPtrT>
    void pre_solve_typed(SolverPtrT&    solver_ptr,
                         PyDenseMatrix& rhs,
                         PyDenseMatrix& solution)
    {
        using T            = typename SolverPtrT::element_type::Type;
        auto& rhs_mat      = std::get<DenseMatrixPtr<T>>(rhs.matrix);
        auto& solution_mat = std::get<DenseMatrixPtr<T>>(solution.matrix);
        solver_ptr->pre_solve(*rhs_mat, *solution_mat);
    }

    template <typename SolverPtrT>
    void solve_typed(SolverPtrT&    solver_ptr,
                     PyDenseMatrix& rhs,
                     PyDenseMatrix& solution)
    {
        using T            = typename SolverPtrT::element_type::Type;
        auto& rhs_mat      = std::get<DenseMatrixPtr<T>>(rhs.matrix);
        auto& solution_mat = std::get<DenseMatrixPtr<T>>(solution.matrix);
        solver_ptr->solve(*rhs_mat, *solution_mat);
    }

    void validate_rhs(const PyDenseMatrix& rhs) const
    {
        if (matrix->dtype() != rhs.dtype()) {
            throw std::invalid_argument(
                "Solver rhs dtype must match the SparseMatrix dtype.");
        }
        if (rhs.rows() != matrix->rows()) {
            throw std::invalid_argument(
                "Solver rhs rows must match SparseMatrix rows.");
        }
        if (rhs.cols() != unknown_dim) {
            throw std::invalid_argument(
                "Solver rhs columns must match solver unknown_dim.");
        }
    }

    void validate_solution(const PyDenseMatrix& solution,
                           const PyDenseMatrix& rhs) const
    {
        if (solution.dtype() != rhs.dtype()) {
            throw std::invalid_argument(
                "Solver solution dtype must match rhs dtype.");
        }
        if (solution.rows() != matrix->cols() ||
            solution.cols() != rhs.cols()) {
            throw std::invalid_argument(
                "Solver solution shape must be (SparseMatrix.cols, rhs.cols).");
        }
    }

    void validate_system(const PyDenseMatrix& rhs,
                         const PyDenseMatrix& solution) const
    {
        validate_rhs(rhs);
        validate_solution(solution, rhs);
    }
};

using PyCGSolver  = PyIterativeSolver<rxmesh::CGSolver>;
using PyPCGSolver = PyIterativeSolver<rxmesh::PCGSolver>;

template <template <typename, int> class SolverT, DirectSolverKind Kind>
struct PyDirectSolver
{
    PyDirectSolver(std::shared_ptr<PySparseMatrix> matrix_in,
                   std::string                     permute)
        : matrix(std::move(matrix_in)),
          permute_method(rxmesh::string_to_permute_method(std::move(permute)))
    {
        if (!matrix) {
            throw std::invalid_argument(
                "Direct solver requires a SparseMatrix.");
        }
        if (matrix->rows() != matrix->cols()) {
            throw std::invalid_argument(
                "Direct solvers require a square SparseMatrix.");
        }

        with_float_double_sparse_matrix(
            *matrix,
            "Direct solvers",
            [&](auto& sparse_mat) {
                using SparseMatT = std::decay_t<decltype(sparse_mat)>;
                using T          = typename SparseMatT::element_type::Type;
                solver = std::make_shared<
                    SolverT<rxmesh::SparseMatrix<T>, Eigen::ColMajor>>(
                    sparse_mat.get(), permute_method);
            });
    }

    std::string name()
    {
        return std::visit([](auto& solver_ptr) { return solver_ptr->name(); },
                          solver);
    }

    std::string permute()
    {
        return rxmesh::permute_method_to_string(permute_method);
    }

    bool is_factorized() const
    {
        return factorized;
    }

    void pre_solve(rxmesh::RXMeshStatic& mesh)
    {
        using namespace rxmesh;
        prepare_matrix_for_device_solver();
        std::visit([&](auto& solver_ptr) { solver_ptr->pre_solve(mesh); },
                   solver);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        factorized = true;
    }

    void solve_into(PyDenseMatrix& rhs,
                    PyDenseMatrix& solution,
                    bool           run_pre_solve)
    {
        using namespace rxmesh;
        validate_system(rhs, solution);

        if constexpr (Kind == DirectSolverKind::LU) {
            prepare_matrix_for_host_solver();
        } else {
            prepare_matrix_for_device_solver();
        }

        if (run_pre_solve && !factorized && matrix->mesh_owner) {
            pre_solve(*matrix->mesh_owner);
        }

        std::visit(
            [&](auto& solver_ptr) { solve_typed(solver_ptr, rhs, solution); },
            solver);

        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        if constexpr (Kind != DirectSolverKind::LU) {
            solution.move(rxmesh::DEVICE, rxmesh::HOST);
        }
    }

    std::shared_ptr<PyDenseMatrix> solve(PyDenseMatrix& rhs,
                                         py::object     initial_guess,
                                         bool           run_pre_solve)
    {
        validate_rhs(rhs);
        auto solution = std::make_shared<PyDenseMatrix>(
            rhs.dtype(),
            matrix->cols(),
            rhs.cols(),
            static_cast<int>(rxmesh::LOCATION_ALL),
            "col_major");

        if (initial_guess.is_none()) {
            solution->reset(py::float_(0.0),
                            static_cast<int>(rxmesh::LOCATION_ALL));
        } else {
            auto initial = initial_guess.cast<std::shared_ptr<PyDenseMatrix>>();
            validate_solution(*initial, rhs);
            solution->copy_from(*initial,
                                static_cast<int>(rxmesh::LOCATION_ALL),
                                static_cast<int>(rxmesh::LOCATION_ALL));
        }

        solve_into(rhs, *solution, run_pre_solve);
        return solution;
    }

    std::shared_ptr<PySparseMatrix> matrix;
    rxmesh::PermuteMethod           permute_method;
    DirectSolverVariant<SolverT>    solver;
    bool                            factorized = false;

   private:
    void prepare_matrix_for_host_solver()
    {
        matrix->ensure_host_readable();
    }

    void prepare_matrix_for_device_solver()
    {
        matrix->ensure_host_readable();
        matrix->ensure_device_readable();
    }

    template <typename SolverPtrT>
    void solve_typed(SolverPtrT&    solver_ptr,
                     PyDenseMatrix& rhs,
                     PyDenseMatrix& solution)
    {
        using T            = typename SolverPtrT::element_type::T;
        auto& rhs_mat      = std::get<DenseMatrixPtr<T>>(rhs.matrix);
        auto& solution_mat = std::get<DenseMatrixPtr<T>>(solution.matrix);

        if constexpr (Kind == DirectSolverKind::Cholesky ||
                      Kind == DirectSolverKind::QR) {
            if (factorized) {
                solver_ptr->solve(*rhs_mat, *solution_mat);
            } else {
                solver_ptr->solve_hl_api(*rhs_mat, *solution_mat);
            }
        } else {
            solver_ptr->solve(*rhs_mat, *solution_mat);
        }
    }

    void validate_rhs(const PyDenseMatrix& rhs) const
    {
        if (matrix->dtype() != rhs.dtype()) {
            throw std::invalid_argument(
                "Direct solver rhs dtype must match the SparseMatrix dtype.");
        }
        if (rhs.rows() != matrix->rows()) {
            throw std::invalid_argument(
                "Direct solver rhs rows must match SparseMatrix rows.");
        }
    }

    void validate_solution(const PyDenseMatrix& solution,
                           const PyDenseMatrix& rhs) const
    {
        if (solution.dtype() != rhs.dtype()) {
            throw std::invalid_argument(
                "Direct solver solution dtype must match rhs dtype.");
        }
        if (solution.rows() != matrix->cols() ||
            solution.cols() != rhs.cols()) {
            throw std::invalid_argument(
                "Direct solver solution shape must be (SparseMatrix.cols, "
                "rhs.cols).");
        }
    }

    void validate_system(const PyDenseMatrix& rhs,
                         const PyDenseMatrix& solution) const
    {
        validate_rhs(rhs);
        validate_solution(solution, rhs);
    }
};

#ifdef USE_CUDSS
template <template <typename, int> class SolverT, DirectSolverKind Kind>
struct PyCuDSSDirectSolver
{
    PyCuDSSDirectSolver(std::shared_ptr<PySparseMatrix> matrix_in,
                        std::string                     permute)
        : matrix(std::move(matrix_in)),
          permute_method(rxmesh::string_to_permute_method(std::move(permute)))
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

        with_float_double_sparse_matrix(
            *matrix,
            "cuDSS direct solvers",
            [&](auto& sparse_mat) {
                using SparseMatT = std::decay_t<decltype(sparse_mat)>;
                using T          = typename SparseMatT::element_type::Type;
                solver = std::make_shared<
                    SolverT<rxmesh::SparseMatrix<T>, Eigen::ColMajor>>(
                    sparse_mat.get(), permute_method);
            });
    }

    std::string name()
    {
        return std::visit([](auto& solver_ptr) { return solver_ptr->name(); },
                          solver);
    }

    std::string permute()
    {
        return rxmesh::permute_method_to_string(permute_method);
    }

    bool is_factorized() const
    {
        return factorized;
    }

    void pre_solve(rxmesh::RXMeshStatic& mesh,
                   PyDenseMatrix&        rhs,
                   PyDenseMatrix&        solution)
    {
        validate_system(rhs, solution);
        matrix->ensure_host_readable();
        matrix->ensure_device_readable();

        std::visit(
            [&](auto& solver_ptr) {
                using SolverPtrT = std::decay_t<decltype(solver_ptr)>;
                using T          = typename SolverPtrT::element_type::T;
                auto& rhs_mat    = std::get<DenseMatrixPtr<T>>(rhs.matrix);
                auto& solution_mat =
                    std::get<DenseMatrixPtr<T>>(solution.matrix);
                solver_ptr->pre_solve(mesh, *rhs_mat, *solution_mat);
            },
            solver);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        factorized = true;
    }

    void solve_into(PyDenseMatrix& rhs,
                    PyDenseMatrix& solution,
                    bool           run_pre_solve)
    {
        validate_system(rhs, solution);
        if (run_pre_solve && !factorized) {
            if (!matrix->mesh_owner) {
                throw std::invalid_argument(
                    "cuDSSCholeskySolver.solve_into(pre_solve=True) needs a "
                    "mesh-owned SparseMatrix or an explicit pre_solve(mesh, "
                    "rhs, solution) call.");
            }
            pre_solve(*matrix->mesh_owner, rhs, solution);
        }
        if (!factorized) {
            throw std::runtime_error(
                "cuDSSCholeskySolver.solve_into() requires pre_solve() before "
                "solve.");
        }

        std::visit(
            [&](auto& solver_ptr) {
                using SolverPtrT = std::decay_t<decltype(solver_ptr)>;
                using T          = typename SolverPtrT::element_type::T;
                auto& rhs_mat    = std::get<DenseMatrixPtr<T>>(rhs.matrix);
                auto& solution_mat =
                    std::get<DenseMatrixPtr<T>>(solution.matrix);
                solver_ptr->solve(*rhs_mat, *solution_mat);
            },
            solver);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));

        solution.move(rxmesh::DEVICE, rxmesh::HOST);
    }

    std::shared_ptr<PyDenseMatrix> solve(PyDenseMatrix& rhs,
                                         py::object     initial_guess,
                                         bool           run_pre_solve)
    {
        validate_rhs(rhs);
        auto solution = std::make_shared<PyDenseMatrix>(
            rhs.dtype(),
            matrix->cols(),
            rhs.cols(),
            static_cast<int>(rxmesh::LOCATION_ALL),
            "col_major");
        if (initial_guess.is_none()) {
            solution->reset(py::float_(0.0),
                            static_cast<int>(rxmesh::LOCATION_ALL));
        } else {
            auto initial = initial_guess.cast<std::shared_ptr<PyDenseMatrix>>();
            validate_solution(*initial, rhs);
            solution->copy_from(*initial,
                                static_cast<int>(rxmesh::LOCATION_ALL),
                                static_cast<int>(rxmesh::LOCATION_ALL));
        }
        solve_into(rhs, *solution, run_pre_solve);
        return solution;
    }

    std::shared_ptr<PySparseMatrix> matrix;
    rxmesh::PermuteMethod           permute_method;
    DirectSolverVariant<SolverT>    solver;
    bool                            factorized = false;

   private:
    void validate_rhs(const PyDenseMatrix& rhs) const
    {
        if (matrix->dtype() != rhs.dtype()) {
            throw std::invalid_argument(
                "cuDSS direct solver rhs dtype must match the SparseMatrix "
                "dtype.");
        }
        if (rhs.rows() != matrix->rows()) {
            throw std::invalid_argument(
                "cuDSS direct solver rhs rows must match SparseMatrix rows.");
        }
    }

    void validate_solution(const PyDenseMatrix& solution,
                           const PyDenseMatrix& rhs) const
    {
        if (solution.dtype() != rhs.dtype()) {
            throw std::invalid_argument(
                "cuDSS direct solver solution dtype must match rhs dtype.");
        }
        if (solution.rows() != matrix->cols() ||
            solution.cols() != rhs.cols()) {
            throw std::invalid_argument(
                "cuDSS direct solver solution shape must be "
                "(SparseMatrix.cols, rhs.cols).");
        }
    }

    void validate_system(const PyDenseMatrix& rhs,
                         const PyDenseMatrix& solution) const
    {
        validate_rhs(rhs);
        validate_solution(solution, rhs);
    }
};
#endif

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

using PyCholeskySolver =
    PyDirectSolver<rxmesh::CholeskySolver, DirectSolverKind::Cholesky>;
using PyQRSolver = PyDirectSolver<rxmesh::QRSolver, DirectSolverKind::QR>;
using PyLUSolver = PyDirectSolver<rxmesh::LUSolver, DirectSolverKind::LU>;
#ifdef USE_CUDSS
using PycuDSSCholeskySolver =
    PyCuDSSDirectSolver<rxmesh::cuDSSCholeskySolver,
                        DirectSolverKind::cuDSSCholesky>;
#endif

}  // namespace pyrxmesh_py
