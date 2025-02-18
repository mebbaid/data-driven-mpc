// data-driven-mpc.cpp

#include "data-driven-mpc.h"
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <iostream>
#include <deque> // Include deque

namespace DataDrivenMPC
{

    DDMPC::DDMPC(int horizonLength, int predictionHorizon, int controlHorizon,
                 const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, QPSolver *solver)
        : m_horizonLength(horizonLength), m_predictionHorizon(predictionHorizon),
          m_controlHorizon(controlHorizon), m_Q(Q), m_R(R), m_solver(solver)
    {

        if (m_controlHorizon > m_predictionHorizon)
        {
            throw std::invalid_argument("Control horizon (M) cannot be greater than the prediction horizon (N).");
        }
        if (Q.rows() != Q.cols() || Q.rows() != predictionHorizon)
        {
            throw std::invalid_argument("Q matrix has the incorrect dimensions. It must me diagonal and N x N, where N is the prediction horizon");
        }
        if (R.rows() != R.cols() || R.rows() != controlHorizon)
        {
            throw std::invalid_argument("R matrix has the incorrect dimensions. It must me diagonal and M x M, where M is the control horizon");
        }
        if (!m_solver)
        {
            throw std::invalid_argument("Solver cannot be a null pointer.");
        }

        // Set up the solver.
        m_solver->instantiateSolver("osqp");                  // Or "proxqp", etc.
        m_solver->setBooleanParameter("warm_starting", true); // Enable warm-starting
    }

    void DDMPC::setInputConstraints(const Eigen::VectorXd &u_min, const Eigen::VectorXd &u_max)
    {
        m_uMin = u_min;
        m_uMax = u_max;
        m_useInputConstraints = true;
    }

    void DDMPC::setDeltaInputConstraints(const Eigen::VectorXd &delta_u_min, const Eigen::VectorXd &delta_u_max)
    {
        m_deltaUmin = delta_u_min;
        m_deltaUmax = delta_u_max;
        m_useDeltaInputConstraints = true;
    }

    void DDMPC::setOutputConstraints(const Eigen::VectorXd &y_min, const Eigen::VectorXd &y_max)
    {
        m_yMin = y_min;
        m_yMax = y_max;
        m_useOutputConstraints = true;
    }

    void DDMPC::updateWeights(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
    {
        if (Q.rows() != Q.cols() || Q.rows() != m_predictionHorizon)
        {
            throw std::invalid_argument("Q matrix has the incorrect dimensions. It must me diagonal and N x N, where N is the prediction horizon");
        }
        if (R.rows() != R.cols() || R.rows() != m_controlHorizon)
        {
            throw std::invalid_argument("R matrix has the incorrect dimensions. It must me diagonal and M x M, where M is the control horizon");
        }

        // Update the weight matrices
        m_Q = Q;
        m_R = R;
    }

    void DDMPC::checkInputData(const std::vector<Eigen::VectorXd> &u_data, const std::vector<Eigen::VectorXd> &y_data) const
    {
        if (u_data.empty() || y_data.empty())
        {
            throw std::invalid_argument("Input/output data vectors cannot be empty.");
        }

        if (u_data.size() != y_data.size())
        {
            throw std::invalid_argument("Input and output data must be the same length.");
        }

        if (u_data.size() < static_cast<size_t>(m_horizonLength))
        {
            throw std::invalid_argument("Insufficient data for the given horizon length.");
        }

        int u_dim = u_data[0].size();
        int y_dim = y_data[0].size();

        // Check the dimensions of each element
        for (size_t i = 0; i < u_data.size(); ++i)
        {
            if (u_data[i].size() != u_dim)
            {
                throw std::invalid_argument("Inconsistent dimensions in input data at index " + std::to_string(i) + ".");
            }
            if (y_data[i].size() != y_dim)
            {
                throw std::invalid_argument("Inconsistent dimensions in output data at index " + std::to_string(i) + ".");
            }
        }
    }

    Eigen::VectorXd DDMPC::solve(const std::vector<Eigen::VectorXd> &u_data_vec,
                                 const std::vector<Eigen::VectorXd> &y_data_vec,
                                 const Eigen::VectorXd &reference,
                                 const Eigen::VectorXd &u_prev)
    {

        if (reference.size() != m_predictionHorizon)
        {
            throw std::invalid_argument("Reference vector size does not match the prediction horizon.");
        }

        std::deque<Eigen::VectorXd> u_data(u_data_vec.begin(), u_data_vec.end());
        std::deque<Eigen::VectorXd> y_data(y_data_vec.begin(), y_data_vec.end());

        HankelMatrix Hu(std::vector<Eigen::VectorXd>(u_data.begin(), u_data.end()), m_horizonLength);
        HankelMatrix Hy(std::vector<Eigen::VectorXd>(y_data.begin(), y_data.end()), m_horizonLength);

        int u_dim = u_data[0].size();
        int y_dim = y_data[0].size();
        Eigen::VectorXd up = Hu.getMatrix().col(Hu.cols() - 1);
        Eigen::VectorXd yp = Hy.getMatrix().col(Hy.cols() - 1);

        int excitation_order = m_horizonLength * u_dim;
        if (!Hu.isPersistentlyExciting(excitation_order))
        {
            std::cerr << "Warning: Input data may not be persistently exciting." << std::endl;
            std::cerr << "Hu rows: " << Hu.rows() << ", Hu cols: " << Hu.cols() << std::endl;
        }

        int num_g = Hu.cols(); // CORRECT CALCULATION OF NUM_G
        Eigen::SparseMatrix<double> H(num_g, num_g);
        Eigen::MatrixXd f(num_g, 1); // Corrected initialization
        buildOptimizationProblem(Hu.getMatrix(), Hy.getMatrix(), up, yp, reference, u_prev, H, f);

        // --- Constraints ---
        int eq_constraints_size = (m_horizonLength + m_predictionHorizon) * (u_dim + y_dim);
        int total_constraints_rows = eq_constraints_size;

        if (m_useInputConstraints)
        {
            total_constraints_rows += m_controlHorizon * u_dim;
        }
        if (m_useOutputConstraints)
        {
            total_constraints_rows += m_predictionHorizon * y_dim;
        }
        if (m_useDeltaInputConstraints)
        {
            total_constraints_rows += m_controlHorizon * u_dim;
        }

        // Declare DENSE matrices for building constraints
        Eigen::MatrixXd A_dense(total_constraints_rows, num_g);
        Eigen::VectorXd lower_bound_dense(total_constraints_rows);
        Eigen::VectorXd upper_bound_dense(total_constraints_rows);
        A_dense.setZero(); // Initialize to zero

        // --- Equality Constraints (a) ---
        // Eigen::MatrixXd combined_hankel(Hu.rows() + Hy.rows(), Hu.cols());
        // combined_hankel << Hu.getMatrix(), Hy.getMatrix();

        // A_dense.block(0, 0, eq_constraints_size, num_g) = combined_hankel.block(0, 0, eq_constraints_size, num_g);
        // lower_bound_dense.head(eq_constraints_size) = combined_hankel.col(combined_hankel.cols() - 1).head(eq_constraints_size);
        // upper_bound_dense.head(eq_constraints_size) = combined_hankel.col(combined_hankel.cols() - 1).head(eq_constraints_size);
        A_dense.block(0, 0, Hu.rows(), num_g) = Hu.getMatrix();
        A_dense.block(Hu.rows(), 0, Hy.rows(), num_g) = Hy.getMatrix();

        lower_bound_dense.head(Hu.rows()) = Hu.getMatrix().col(Hu.cols() - 1);
        lower_bound_dense.segment(Hu.rows(), Hy.rows()) = Hy.getMatrix().col(Hy.cols() - 1);
        upper_bound_dense.head(Hu.rows()) = Hu.getMatrix().col(Hu.cols() - 1);
        upper_bound_dense.segment(Hu.rows(), Hy.rows()) = Hy.getMatrix().col(Hy.cols() - 1);

        int constraint_row_index = Hu.rows() + Hy.rows();

        // --- Input Constraints (b) ---
        if (m_useInputConstraints)
        {
            Eigen::MatrixXd U(m_controlHorizon * u_dim, num_g);
            U.setZero();
            Eigen::MatrixXd Up = Hu.getMatrix().block((m_horizonLength - m_controlHorizon) * u_dim, 0, m_controlHorizon * u_dim, Hu.cols());
            U.noalias() = Up;

            A_dense.block(constraint_row_index, 0, m_controlHorizon * u_dim, num_g) = U;
            for (int i = 0; i < m_controlHorizon * u_dim; ++i)
            {
                lower_bound_dense(constraint_row_index + i) = m_uMin(i % u_dim);
                upper_bound_dense(constraint_row_index + i) = m_uMax(i % u_dim);
            }

            constraint_row_index += m_controlHorizon * u_dim;
        }

        // --- Output Constraints (c) ---
        if (m_useOutputConstraints)
        {
            Eigen::MatrixXd Y(m_predictionHorizon * y_dim, num_g);
            Y.setZero();
            Eigen::MatrixXd Yf = Hy.getMatrix().block((m_horizonLength - m_predictionHorizon) * y_dim, 0, m_predictionHorizon * y_dim, Hy.cols());
            Y.noalias() = Yf;

            A_dense.block(constraint_row_index, 0, m_predictionHorizon * y_dim, num_g) = Y;

            for (int i = 0; i < m_predictionHorizon * y_dim; ++i)
            {
                lower_bound_dense(constraint_row_index + i) = m_yMin(i % y_dim);
                upper_bound_dense(constraint_row_index + i) = m_yMax(i % y_dim);
            }
            constraint_row_index += m_predictionHorizon * y_dim;
        }

        if (m_useDeltaInputConstraints)
        {
            Eigen::MatrixXd U(m_controlHorizon * u_dim, num_g);
            U.setZero();
            Eigen::MatrixXd Up = Hu.getMatrix().block((m_horizonLength - m_controlHorizon) * u_dim, 0, m_controlHorizon * u_dim, Hu.cols());
            U.noalias() = Up;

            for (int i = 0; i < m_controlHorizon * u_dim; i += u_dim)
            {
                for (int j = 0; j < num_g; ++j)
                {
                    double value = U(i, j); // Get u(k)
                    if (i >= u_dim)         // if not the first u
                    {
                        value -= U(i - u_dim, j); // u(k) - u(k-1)
                    }
                    else
                    {
                        value -= up(up.rows() - u_dim + (i % u_dim));
                    }
                    // A_dense(constraint_row_index + i, j) = value;
                    A_dense(constraint_row_index + i / u_dim, j) = value; // we fill A row by row, and each row has num_g columns.
                }

                for (int k = 0; k < u_dim; k++)
                {
                    lower_bound_dense(constraint_row_index + i / u_dim + k) = m_deltaUmin(k); // we fill element by element
                    upper_bound_dense(constraint_row_index + i / u_dim + k) = m_deltaUmax(k);
                }
            }
            constraint_row_index += m_controlHorizon;
        }

        // Convert A_dense to sparse A (AFTER building constraints)
        Eigen::SparseMatrix<double> A = A_dense.sparseView();

        // --- Solve the QP ---
        m_solver->setNumberOfVariables(num_g);
        m_solver->setNumberOfConstraints(total_constraints_rows);
        m_solver->setHessianMatrix(H);
        m_solver->setGradient(f);
        m_solver->setLinearConstraintsMatrix(A);
        m_solver->setLowerBound(lower_bound_dense);
        m_solver->setUpperBound(upper_bound_dense);

        if (!m_solver->initSolver())
        {
            throw std::runtime_error("Failed to initialize QP solver.");
        }

        if (m_solver->solveProblem() != QpSolversEigen::ErrorExitFlag::NoError)
        {
            throw std::runtime_error("QP solver failed to find a solution.");
        }

        Eigen::VectorXd g_optimal = m_solver->getSolution();
        Eigen::MatrixXd U_first(u_dim, num_g);
        U_first.setZero();
        // Get the first block of Uf
        U_first.noalias() = Hu.getMatrix().block((m_horizonLength - 1) * u_dim, 0, u_dim, Hu.cols());
        Eigen::VectorXd u_first = U_first * g_optimal;

        return u_first;
    }

    void DDMPC::buildOptimizationProblem(const Eigen::MatrixXd &Hu, const Eigen::MatrixXd &Hy,
                                         const Eigen::VectorXd &up, const Eigen::VectorXd &yp,
                                         const Eigen::VectorXd &reference, const Eigen::VectorXd &u_prev,
                                         Eigen::SparseMatrix<double> &H,
                                         Eigen::MatrixXd &f)
    {
        int u_dim = up.size() / m_horizonLength;
        int y_dim = yp.size() / m_horizonLength;
        int num_g = Hu.cols();

        // Construct Yf and Uf.
        Eigen::MatrixXd Yf(m_predictionHorizon * y_dim, num_g);
        Eigen::MatrixXd Uf(m_controlHorizon * u_dim, num_g);
        Yf.setZero();
        Uf.setZero();


        // Extract Yf and Uf_complete.
        Yf.noalias() = Hy.block((m_horizonLength - m_predictionHorizon) * y_dim, 0, m_predictionHorizon * y_dim, num_g);
        Eigen::MatrixXd Uf_complete = Hu.block((m_horizonLength - m_predictionHorizon) * u_dim, 0, m_predictionHorizon * u_dim, num_g);
        Uf.noalias() = Uf_complete.block(0, 0, m_controlHorizon * u_dim, num_g);

        // Expand Q and R. TODO: handle MIMO case
        Eigen::MatrixXd Q_expanded(m_predictionHorizon * y_dim, m_predictionHorizon * y_dim);
        Eigen::MatrixXd R_expanded(m_controlHorizon * u_dim, m_controlHorizon * u_dim);

        Q_expanded.setIdentity();
        Q_expanded *= m_Q(0, 0);

        R_expanded.setIdentity();
        R_expanded *= m_R(0, 0);

        // construct H and f
        Eigen::MatrixXd H_dense = Yf.transpose() * Q_expanded * Yf + Uf.transpose() * R_expanded * Uf;
        Eigen::VectorXd ref_expanded = reference.replicate(y_dim, 1);
        f = -2 * Yf.transpose() * Q_expanded * ref_expanded;
    }
} // namespace DataDrivenMPC
