#include "data-driven-mpc.h"
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <iostream>
#include <deque>

namespace DataDrivenMPC {

DDMPC::DDMPC(int Tini, int predictionHorizon, int controlHorizon,
             const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, QPSolver* solver)
    : m_Tini(Tini), m_predictionHorizon(predictionHorizon),
      m_controlHorizon(controlHorizon), m_Q(Q), m_R(R), m_solver(solver) {

    if (m_controlHorizon > m_predictionHorizon) {
        throw std::invalid_argument("Control horizon (M) cannot be greater than the prediction horizon (N).");
    }
    if (Q.rows() != Q.cols() || Q.rows() != predictionHorizon) {
        throw std::invalid_argument("Q matrix has incorrect dimensions. It must be diagonal and N x N.");
    }
    if (R.rows() != R.cols() || R.rows() != controlHorizon) {
        throw std::invalid_argument("R matrix has incorrect dimensions. It must be diagonal and M x M.");
    }
    if (!m_solver) {
        throw std::invalid_argument("Solver cannot be a null pointer.");
    }

    m_solver->instantiateSolver("osqp");
    m_solver->setBooleanParameter("warm_starting", true);
}
void DDMPC::setInputConstraints(const Eigen::VectorXd& u_min, const Eigen::VectorXd& u_max)
{
    m_uMin = u_min;
    m_uMax = u_max;
    m_useInputConstraints = true;
}

void DDMPC::setDeltaInputConstraints(const Eigen::VectorXd& delta_u_min, const Eigen::VectorXd& delta_u_max)
{
    m_deltaUmin = delta_u_min;
    m_deltaUmax = delta_u_max;
    m_useDeltaInputConstraints = true;
}

void DDMPC::setOutputConstraints(const Eigen::VectorXd& y_min, const Eigen::VectorXd& y_max)
{
    m_yMin = y_min;
    m_yMax = y_max;
    m_useOutputConstraints = true;
}

void DDMPC::updateWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R)
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

void DDMPC::checkInputData(const std::vector<Eigen::VectorXd>& u_data, const std::vector<Eigen::VectorXd>& y_data) const
{
 if (u_data.empty() || y_data.empty())
    {
        throw std::invalid_argument("Input/output data vectors cannot be empty.");
    }

    if (u_data.size() != y_data.size())
    {
        throw std::invalid_argument("Input and output data must be the same length.");
    }

    if (u_data.size() < static_cast<size_t>(m_Tini + m_predictionHorizon))
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

Eigen::VectorXd DDMPC::solve(const std::vector<Eigen::VectorXd>& u_data_vec,
                             const std::vector<Eigen::VectorXd>& y_data_vec,
                             const Eigen::VectorXd& reference,
                             const Eigen::VectorXd& u_prev) {

    if (reference.size() != m_predictionHorizon) {
        throw std::invalid_argument("Reference vector size does not match the prediction horizon.");
    }

    // --- KEY CHANGE: Use a consistent number of columns for Hankel matrices ---
    static int max_num_g = u_data_vec.size() - (m_Tini + m_predictionHorizon) + 1;
    int num_g = u_data_vec.size() - (m_Tini + m_predictionHorizon) + 1;
    int u_dim = u_data_vec[0].size();
    int y_dim = y_data_vec[0].size();

    // 1. Create Full Hankel Matrices
    HankelMatrix Hu(u_data_vec, m_Tini + m_predictionHorizon);
    HankelMatrix Hy(y_data_vec, m_Tini + m_predictionHorizon);

     // Check persistent excitation
        int excitation_order = (m_Tini + m_predictionHorizon) * u_dim;
        if (!Hu.isPersistentlyExciting(excitation_order, 1e-3))
        {
            std::cerr << "Warning: Input data may not be persistently exciting." << std::endl;
            std::cerr << "Hu rows: " << Hu.rows() << ", Hu cols: " << Hu.cols() << std::endl;
        }

    // 2. Split into Past and Future
    Eigen::MatrixXd Up = Hu.getMatrix().block(0, 0, m_Tini * u_dim, num_g);
    Eigen::MatrixXd Yp = Hy.getMatrix().block(0, 0, m_Tini * y_dim, num_g);
    Eigen::MatrixXd Uf = Hu.getMatrix().block(m_Tini * u_dim, 0, m_predictionHorizon * u_dim, num_g);
    Eigen::MatrixXd Yf = Hy.getMatrix().block(m_Tini * y_dim, 0, m_predictionHorizon * y_dim, num_g);

    // 3. Create u_ini and y_ini
    Eigen::VectorXd u_ini(m_Tini * u_dim);
    Eigen::VectorXd y_ini(m_Tini * y_dim);
    for (int i = 0; i < m_Tini; ++i) {
        u_ini.segment(i * u_dim, u_dim) = u_data_vec[i];
        y_ini.segment(i * y_dim, y_dim) = y_data_vec[i];
    }

    // 4. Build the QP Problem

    // 4.1 Cost Function (H and f)
    Eigen::MatrixXd Q_expanded = Eigen::MatrixXd::Identity(m_predictionHorizon * y_dim, m_predictionHorizon* y_dim);
    Eigen::MatrixXd R_expanded = Eigen::MatrixXd::Identity(m_controlHorizon * u_dim, m_controlHorizon * u_dim);
    // std::cout << "Q_expanded size: " << Q_expanded.rows() << " x " << Q_expanded.cols() << std::endl;
    // std::cout << "R_expanded size: " << R_expanded.rows() << " x " << R_expanded.cols() << std::endl;
    // std::cout << "m_Q size: " << m_Q.rows() << " x " << m_Q.cols() << std::endl;
    // std::cout << "m_R size: " << m_R.rows() << " x " << m_R.cols() << std::endl;
    // for (int i=0; i< m_predictionHorizon; ++i){
    //     Q_expanded.block(i*y_dim,i*y_dim,y_dim,y_dim) = m_Q;
    // }
    //  for (int i=0; i< m_controlHorizon; ++i){
    //     R_expanded.block(i*u_dim,i*u_dim,u_dim,u_dim) = m_R;
    // }
    Q_expanded = m_Q;
    R_expanded = m_R;
    std::cout << "Q_expanded size: " << Q_expanded.rows() << " x " << Q_expanded.cols() << std::endl;
    std::cout << "R_expanded size: " << R_expanded.rows() << " x " << R_expanded.cols() << std::endl;
    std::cout << "Yf.transpose() size: " << Yf.transpose().rows() << " x " << Yf.transpose().cols() << std::endl;
    std::cout << "Uf.transpose() size: " << Uf.transpose().rows() << " x " << Uf.transpose().cols() << std::endl;
    Eigen::MatrixXd H_dense = Yf.transpose() * Q_expanded * Yf + Uf.transpose() * R_expanded * Uf;
    Eigen::SparseMatrix<double> H = H_dense.sparseView();

    Eigen::VectorXd ref_expanded = reference.replicate(y_dim, 1);
    Eigen::MatrixXd f = -2 * Yf.transpose() * Q_expanded * ref_expanded;
    // --- 4.2 Constraints ---

    // Calculate *initial* total number of constraints
    int total_constraints_rows = (m_Tini + m_predictionHorizon) * (u_dim + y_dim);


    // Declare matrices/vectors for building constraints
    Eigen::MatrixXd A_dense(total_constraints_rows, max_num_g); // max_num_g columns
    Eigen::VectorXd lower_bound_dense(total_constraints_rows);
    Eigen::VectorXd upper_bound_dense(total_constraints_rows);
    A_dense.setZero(); // Initialize to zero

    // --- Equality Constraints ---
    Eigen::MatrixXd A_eq( (m_Tini + m_predictionHorizon) * (u_dim + y_dim) , num_g);
    A_eq << Up, Yp, Uf, Yf;
    A_dense.block(0, 0, A_eq.rows(), num_g) = A_eq;

    Eigen::VectorXd b_eq( (m_Tini + m_predictionHorizon) * (u_dim + y_dim));
    b_eq << u_ini, y_ini, Eigen::VectorXd::Zero((m_predictionHorizon) * (u_dim + y_dim));
    lower_bound_dense.head(b_eq.size()) = b_eq;
    upper_bound_dense.head(b_eq.size()) = b_eq;


    int constraint_row_index = A_eq.rows();

    // --- Inequality Constraints (with dynamic row count) ---
   // Input Constraints
    if (m_useInputConstraints) {
        // Recalculate total_constraints_rows *before* assignment
        total_constraints_rows = constraint_row_index + m_controlHorizon * u_dim;
        A_dense.conservativeResize(total_constraints_rows, max_num_g); // Resize A_dense

        Eigen::MatrixXd A_u = Uf.block(0, 0, m_controlHorizon * u_dim, num_g);
        A_dense.block(constraint_row_index, 0, A_u.rows(), num_g) = A_u;

        Eigen::VectorXd l_u(m_controlHorizon * u_dim);
        Eigen::VectorXd u_u(m_controlHorizon * u_dim);

        for (int i = 0; i < m_controlHorizon; i++) {
          l_u.segment(i*u_dim,u_dim) = m_uMin;
          u_u.segment(i*u_dim,u_dim) = m_uMax;
        }
        lower_bound_dense.conservativeResize(total_constraints_rows); // Resize bounds
        upper_bound_dense.conservativeResize(total_constraints_rows); // Resize bounds
        lower_bound_dense.segment(constraint_row_index, l_u.size()) = l_u;
        upper_bound_dense.segment(constraint_row_index, u_u.size()) = u_u;


        constraint_row_index += A_u.rows();
    }

    // Output Constraints
    if (m_useOutputConstraints) {
        // Recalculate total_constraints_rows *before* assignment
        total_constraints_rows = constraint_row_index + m_predictionHorizon * y_dim;
        A_dense.conservativeResize(total_constraints_rows, max_num_g); // Resize A_dense

        Eigen::MatrixXd A_y = Yf;
        A_dense.block(constraint_row_index, 0, A_y.rows(), num_g) = A_y;

        Eigen::VectorXd l_y(m_predictionHorizon * y_dim);
        Eigen::VectorXd u_y(m_predictionHorizon * y_dim);
        for(int i = 0; i<m_predictionHorizon; i++){
          l_y.segment(i*y_dim, y_dim) = m_yMin;
          u_y.segment(i*y_dim, y_dim) = m_yMax;
        }
        lower_bound_dense.conservativeResize(total_constraints_rows); // Resize bounds
        upper_bound_dense.conservativeResize(total_constraints_rows); // Resize bounds
        lower_bound_dense.segment(constraint_row_index, l_y.size()) = l_y;
        upper_bound_dense.segment(constraint_row_index, u_y.size()) = u_y;


        constraint_row_index += A_y.rows();
    }

    // Delta Input Constraints
     if (m_useDeltaInputConstraints) {
        // Recalculate total_constraints_rows *before* assignment
        total_constraints_rows = constraint_row_index + m_controlHorizon * u_dim;
        A_dense.conservativeResize(total_constraints_rows, max_num_g); // Resize A_dense

        Eigen::MatrixXd A_delta_u(m_controlHorizon * u_dim, num_g);
        A_delta_u.setZero();

        // Construct the matrix that calculates delta_u
        for (int i = 0; i < m_controlHorizon; ++i)
        {
            for (int j = 0; j < num_g; ++j)
            {
                // Current u(k)
                A_delta_u(i*u_dim,j) = Uf(i * u_dim, j);

                // Subtract u(k-1)
                if (i > 0)
                {
                  A_delta_u(i*u_dim,j) -= Uf((i - 1) * u_dim, j);
                }
                else // For the first control input, subtract the last element of u_ini
                {
                  A_delta_u(i*u_dim, j) -= u_ini.segment((m_Tini - 1) * u_dim, u_dim)(j%u_dim);
                }
            }
        }

        A_dense.block(constraint_row_index, 0, A_delta_u.rows(), num_g) = A_delta_u;

        Eigen::VectorXd l_delta_u(m_controlHorizon * u_dim);
        Eigen::VectorXd u_delta_u(m_controlHorizon * u_dim);
        for(int i = 0; i < m_controlHorizon; i++){
          l_delta_u.segment(i*u_dim, u_dim) = m_deltaUmin;
          u_delta_u.segment(i*u_dim, u_dim) = m_deltaUmax;
        }

        lower_bound_dense.conservativeResize(total_constraints_rows); // Resize bounds
        upper_bound_dense.conservativeResize(total_constraints_rows); // Resize bounds
        lower_bound_dense.segment(constraint_row_index, l_delta_u.size()) = l_delta_u;
        upper_bound_dense.segment(constraint_row_index, u_delta_u.size()) = u_delta_u;

        constraint_row_index += A_delta_u.rows(); // Important: Use correct increment
    }


    // --- 5. Solve the QP ---

    Eigen::SparseMatrix<double> A = A_dense.sparseView();

    m_solver->setNumberOfVariables(max_num_g);  // max_num_g variables
    m_solver->setNumberOfConstraints(total_constraints_rows); // constraints
    m_solver->setHessianMatrix(H);
    m_solver->setGradient(f);
    m_solver->setLinearConstraintsMatrix(A);
    m_solver->setLowerBound(lower_bound_dense);
    m_solver->setUpperBound(upper_bound_dense);

    if (!m_solver->initSolver()) {
        throw std::runtime_error("Failed to initialize QP solver.");
    }
    if (m_solver->solveProblem() != QpSolversEigen::ErrorExitFlag::NoError) {
        throw std::runtime_error("QP solver failed to find a solution.");
    }

    // 6. Extract and Return the First Control Input
    Eigen::VectorXd g_optimal = m_solver->getSolution(); // g optimal has max_num_g dimensions
    Eigen::MatrixXd U_first_block = Uf.block(0, 0, u_dim, num_g); // Get first block of Uf
    Eigen::VectorXd u_first = U_first_block * g_optimal;
    return u_first;
}

} // namespace DataDrivenMPC