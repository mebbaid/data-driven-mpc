#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "data-driven-mpc/data-driven-mpc.h" // Make sure path is correct
#include <memory> // For std::unique_ptr, std::make_unique

namespace py = pybind11;

// Removed createQPSolver as we'll create it internally now

PYBIND11_MODULE(py_ddmpc, m) {
    m.doc() = "Python bindings for DataDrivenMPC (with scaling)";

    py::class_<DataDrivenMPC::HankelMatrix>(m, "HankelMatrix")
        // Keep HankelMatrix bindings as they are (assuming they were correct)
        .def(py::init<const std::vector<double> &, int>()) // Example, adjust if needed
        .def(py::init<const std::vector<Eigen::VectorXd> &, int>())
        .def("get_matrix", &DataDrivenMPC::HankelMatrix::getMatrix)
        .def("is_persistently_exciting", &DataDrivenMPC::HankelMatrix::isPersistentlyExciting,
             py::arg("order"), py::arg("tolerance") = 1e-6)
        .def("get_horizon_length", &DataDrivenMPC::HankelMatrix::getHorizonLength)
        .def("rows", &DataDrivenMPC::HankelMatrix::rows)
        .def("cols", &DataDrivenMPC::HankelMatrix::cols);

    // QPSolver is now an internal detail of DDMPC binding, don't expose it unless needed elsewhere
    // py::class_<DataDrivenMPC::QPSolver, std::shared_ptr<DataDrivenMPC::QPSolver>>(m, "QPSolver");

    py::class_<DataDrivenMPC::DDMPC>(m, "DDMPC")
        // Update the constructor binding
        .def(py::init([](int Tini, int predictionHorizon, int controlHorizon,
                         const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, std::string solver_name,
                         double lambda_u, double lambda_y, double lambda_g,
                         // Add new scaling arguments to Python signature
                         double scale_g, double scale_u, double scale_y)
                       {
                            // Create the QPSolver internally as a unique_ptr
                            auto solver_ptr = std::make_unique<DataDrivenMPC::QPSolver>();
                            // Call the C++ constructor, moving the solver pointer
                            // and passing the new scaling arguments
                            return std::make_unique<DataDrivenMPC::DDMPC>(
                                Tini, predictionHorizon, controlHorizon, Q, R,
                                std::move(solver_ptr), solver_name,
                                lambda_u, lambda_y, lambda_g,
                                scale_g, scale_u, scale_y // Pass new arguments
                            );
                       }),
             // Update the py::arg list
             py::arg("Tini"), py::arg("predictionHorizon"), py::arg("controlHorizon"),
             py::arg("Q"), py::arg("R"),
             py::arg("solver_name"), // Default solver name
             // Solver argument removed from Python args
             py::arg("lambda_u"), // Keep lambda args mandatory as before
             py::arg("lambda_y"),
             py::arg("lambda_g"),
             // Add py::arg for new scaling factors, provide defaults for convenience
             py::arg("scale_g") = 1.0,
             py::arg("scale_u") = 1.0,
             py::arg("scale_y") = 1.0
            )
        // Keep other method bindings the same
        .def("set_input_constraints", &DataDrivenMPC::DDMPC::setInputConstraints)
        .def("set_output_constraints", &DataDrivenMPC::DDMPC::setOutputConstraints)
        .def("set_delta_input_constraints", &DataDrivenMPC::DDMPC::setDeltaInputConstraints)
        .def("set_lambda_g", &DataDrivenMPC::DDMPC::setLambdaG) // Note: Check if setLambdaU/Y are needed
        .def("update_weights", py::overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXd&>(&DataDrivenMPC::DDMPC::updateWeights), "Update weights with single matrices")
        .def("update_weights", py::overload_cast<const std::vector<Eigen::MatrixXd>&, const std::vector<Eigen::MatrixXd>&>(&DataDrivenMPC::DDMPC::updateWeights), "Update weights with vectors of matrices")
        .def("get_lambda_u", &DataDrivenMPC::DDMPC::getLambdaU)
        .def("get_lambda_y", &DataDrivenMPC::DDMPC::getLambdaY)
        .def("get_lambda_g", &DataDrivenMPC::DDMPC::getLambdaG)
        .def("update_regularization_weights", &DataDrivenMPC::DDMPC::updateRegularizationWeights)
        .def("solve", &DataDrivenMPC::DDMPC::solve);

    // Removed create_qp_solver as it's no longer needed by the DDMPC constructor binding
    // m.def("create_qp_solver", &createQPSolver, "Creates a QPSolver instance.");
}