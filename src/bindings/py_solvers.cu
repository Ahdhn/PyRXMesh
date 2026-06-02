#include "bindings/py_solvers.h"

namespace pyrxmesh_py {

void register_solvers(py::module_& m)
{
    py::class_<PyCGSolver, std::shared_ptr<PyCGSolver>>(m, "CGSolver")
        .def(py::init<std::shared_ptr<PySparseMatrix>,
                      int,
                      int,
                      py::object,
                      py::object,
                      int>(),
             py::arg("matrix"),
             py::arg("unknown_dim")         = 1,
             py::arg("max_iter")            = 1000,
             py::arg("abs_tol")             = py::float_(1e-6),
             py::arg("rel_tol")             = py::float_(0.0),
             py::arg("reset_residual_freq") = std::numeric_limits<int>::max())
        .def_property_readonly("name", &PyCGSolver::name)
        .def_property_readonly("iter_taken", &PyCGSolver::iter_taken)
        .def_property_readonly("start_residual", &PyCGSolver::start_residual)
        .def_property_readonly("final_residual", &PyCGSolver::final_residual)
        .def("pre_solve",
             &PyCGSolver::pre_solve,
             py::arg("rhs"),
             py::arg("solution"))
        .def("solve_into",
             &PyCGSolver::solve_into,
             py::arg("rhs"),
             py::arg("solution"),
             py::arg("pre_solve") = true)
        .def("solve",
             &PyCGSolver::solve,
             py::arg("rhs"),
             py::arg("initial_guess") = py::none(),
             py::arg("pre_solve")     = true);

    py::class_<PyPCGSolver, std::shared_ptr<PyPCGSolver>>(m, "PCGSolver")
        .def(py::init<std::shared_ptr<PySparseMatrix>,
                      int,
                      int,
                      py::object,
                      py::object,
                      int>(),
             py::arg("matrix"),
             py::arg("unknown_dim")         = 1,
             py::arg("max_iter")            = 1000,
             py::arg("abs_tol")             = py::float_(1e-6),
             py::arg("rel_tol")             = py::float_(0.0),
             py::arg("reset_residual_freq") = std::numeric_limits<int>::max())
        .def_property_readonly("name", &PyPCGSolver::name)
        .def_property_readonly("iter_taken", &PyPCGSolver::iter_taken)
        .def_property_readonly("start_residual", &PyPCGSolver::start_residual)
        .def_property_readonly("final_residual", &PyPCGSolver::final_residual)
        .def("pre_solve",
             &PyPCGSolver::pre_solve,
             py::arg("rhs"),
             py::arg("solution"))
        .def("solve_into",
             &PyPCGSolver::solve_into,
             py::arg("rhs"),
             py::arg("solution"),
             py::arg("pre_solve") = true)
        .def("solve",
             &PyPCGSolver::solve,
             py::arg("rhs"),
             py::arg("initial_guess") = py::none(),
             py::arg("pre_solve")     = true);

    py::class_<PyCholeskySolver, std::shared_ptr<PyCholeskySolver>>(
        m, "CholeskySolver")
        .def(py::init<std::shared_ptr<PySparseMatrix>, std::string>(),
             py::arg("matrix"),
             py::arg("permute") = "none")
        .def_property_readonly("name", &PyCholeskySolver::name)
        .def_property_readonly("permute", &PyCholeskySolver::permute)
        .def_property_readonly("is_factorized",
                               &PyCholeskySolver::is_factorized)
        .def("pre_solve", &PyCholeskySolver::pre_solve, py::arg("mesh"))
        .def("solve_into",
             &PyCholeskySolver::solve_into,
             py::arg("rhs"),
             py::arg("solution"),
             py::arg("pre_solve") = false)
        .def("solve",
             &PyCholeskySolver::solve,
             py::arg("rhs"),
             py::arg("initial_guess") = py::none(),
             py::arg("pre_solve")     = false);

    py::class_<PyQRSolver, std::shared_ptr<PyQRSolver>>(m, "QRSolver")
        .def(py::init<std::shared_ptr<PySparseMatrix>, std::string>(),
             py::arg("matrix"),
             py::arg("permute") = "none")
        .def_property_readonly("name", &PyQRSolver::name)
        .def_property_readonly("permute", &PyQRSolver::permute)
        .def_property_readonly("is_factorized", &PyQRSolver::is_factorized)
        .def("pre_solve", &PyQRSolver::pre_solve, py::arg("mesh"))
        .def("solve_into",
             &PyQRSolver::solve_into,
             py::arg("rhs"),
             py::arg("solution"),
             py::arg("pre_solve") = false)
        .def("solve",
             &PyQRSolver::solve,
             py::arg("rhs"),
             py::arg("initial_guess") = py::none(),
             py::arg("pre_solve")     = false);

    py::class_<PyLUSolver, std::shared_ptr<PyLUSolver>>(m, "LUSolver")
        .def(py::init<std::shared_ptr<PySparseMatrix>, std::string>(),
             py::arg("matrix"),
             py::arg("permute") = "none")
        .def_property_readonly("name", &PyLUSolver::name)
        .def_property_readonly("permute", &PyLUSolver::permute)
        .def_property_readonly("is_factorized", &PyLUSolver::is_factorized)
        .def("pre_solve", &PyLUSolver::pre_solve, py::arg("mesh"))
        .def("solve_into",
             &PyLUSolver::solve_into,
             py::arg("rhs"),
             py::arg("solution"),
             py::arg("pre_solve") = false)
        .def("solve",
             &PyLUSolver::solve,
             py::arg("rhs"),
             py::arg("initial_guess") = py::none(),
             py::arg("pre_solve")     = false);

    m.attr("has_cudss") = py::bool_(
#ifdef USE_CUDSS
        true
#else
        false
#endif
    );

#ifdef USE_CUDSS
    py::class_<PycuDSSCholeskySolver, std::shared_ptr<PycuDSSCholeskySolver>>(
        m, "cuDSSCholeskySolver")
        .def(py::init<std::shared_ptr<PySparseMatrix>, std::string>(),
             py::arg("matrix"),
             py::arg("permute") = "none")
        .def_property_readonly("name", &PycuDSSCholeskySolver::name)
        .def_property_readonly("permute", &PycuDSSCholeskySolver::permute)
        .def_property_readonly("is_factorized",
                               &PycuDSSCholeskySolver::is_factorized)
        .def("pre_solve",
             &PycuDSSCholeskySolver::pre_solve,
             py::arg("mesh"),
             py::arg("rhs"),
             py::arg("solution"))
        .def("solve_into",
             &PycuDSSCholeskySolver::solve_into,
             py::arg("rhs"),
             py::arg("solution"),
             py::arg("pre_solve") = true)
        .def("solve",
             &PycuDSSCholeskySolver::solve,
             py::arg("rhs"),
             py::arg("initial_guess") = py::none(),
             py::arg("pre_solve")     = true);
#else
    py::class_<PyUnavailableCuDSSCholeskySolver,
               std::shared_ptr<PyUnavailableCuDSSCholeskySolver>>(
        m, "cuDSSCholeskySolver")
        .def(py::init<std::shared_ptr<PySparseMatrix>, std::string>(),
             py::arg("matrix"),
             py::arg("permute") = "none");
#endif
}

}  // namespace pyrxmesh_py
