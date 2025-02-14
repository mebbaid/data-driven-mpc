#include "data-driven-mpc.h"
#include "osqp-solver.h"

#include <iostream> // For debugging (optional, can be removed later)
#include <cstdlib>  // For std::rand

#include <osqp.h>
#include <types.h>
#include <util.h>

namespace DataDrivenMPC
{

    DDMPC::DDMPC(int horizonLength, int predictionHorizon, int controlHorizon,
                 const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, QPSolver *solver)
        : m_horizonLength(horizonLength), m_predictionHorizon(predictionHorizon),
          m_controlHorizon(controlHorizon), m_Q(Q), m_R(R), m_solver(solver)
    {

        if (horizonLength <= 0 || predictionHorizon <= 0 || controlHorizon <= 0)
        {
            throw std::invalid_argument("Horizons must be positive.");
        }
        if (controlHorizon > predictionHorizon)
        {
            throw std::invalid_argument("Control horizon cannot be greater than prediction horizon.");
        }
        // We assume Q and R are diagonal, only the diagonal is relevant for the cost.
        if (Q.rows() != Q.cols())
        { // Q and R must be square
            throw std::invalid_argument("Q matrix has incorrect dimensions.");
        }
        if (R.rows() != R.cols())
        {
            throw std::invalid_argument("R matrix has incorrect dimensions.");
        }
        if (!solver)
        {
            throw std::invalid_argument("QP solver pointer cannot be null.");
        }
    }

    void DDMPC::setInputConstraints(const Eigen::VectorXd &u_min, const Eigen::VectorXd &u_max)
    {
        if (u_min.size() != u_max.size())
        {
            throw std::invalid_argument("Input contraint vectors (u_min and u_max) must have the same dimension");
        }

        m_uMin = u_min;
        m_uMax = u_max;
        m_useInputConstraints = true;
    }

    void DDMPC::setDeltaInputConstraints(const Eigen::VectorXd &delta_u_min, const Eigen::VectorXd &delta_u_max)
    {
        if (delta_u_min.size() != delta_u_max.size())
        {
            throw std::invalid_argument("Input contraint vectors (delta_u_min and delta_u_max) must have the same dimension");
        }

        m_deltaUmin = delta_u_min;
        m_deltaUmax = delta_u_max;
        m_useDeltaInputConstraints = true;
    }

    void DDMPC::setOutputConstraints(const Eigen::VectorXd &y_min, const Eigen::VectorXd &y_max)
    {

        if (y_min.size() != y_max.size())
        {
            throw std::invalid_argument("Output contraint vectors (y_min and y_max) must have the same dimension");
        }

        m_yMin = y_min;
        m_yMax = y_max;
        m_useOutputConstraints = true;
    }

    void DDMPC::updateWeights(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
    {
        if (Q.rows() != Q.cols() || Q.rows() != m_Q.rows())
        {
            throw std::invalid_argument("Q matrix has incorrect dimensions.");
        }
        if (R.rows() != R.cols() || R.rows() != m_R.rows())
        {
            throw std::invalid_argument("R matrix has incorrect dimensions.");
        }

        m_Q = Q;
        m_R = R;
    }

    Eigen::VectorXd DDMPC::solve(const std::vector<Eigen::VectorXd> &u_data,
                                 const std::vector<Eigen::VectorXd> &y_data,
                                 const Eigen::VectorXd &reference,
                                 const Eigen::VectorXd &u_prev)
    {
        // Check input data
        checkInputData(u_data, y_data);

        // Check reference
        if (reference.size() != m_predictionHorizon * y_data[0].size()) // Use y_data to get outputDim
        {
            throw std::invalid_argument("Reference vector has incorrect dimensions. Expected " + std::to_string(m_predictionHorizon * y_data[0].size()) + " got " + std::to_string(reference.size()));
        }

        // Construct Hankel matrices
        DataDrivenMPC::HankelMatrix Hu(u_data, m_horizonLength);
        DataDrivenMPC::HankelMatrix Hy(y_data, m_horizonLength);

        // Prepare past input/output vectors (up, yp)
        //  up is the stacked vector of the past L inputs
        //  yp is the stacked vector of the past L outputs
        int inputDim = u_data[0].rows();
        int outputDim = y_data[0].rows();
        Eigen::VectorXd up(m_horizonLength * inputDim);
        Eigen::VectorXd yp(m_horizonLength * outputDim);

        for (int i = 0; i < m_horizonLength; ++i)
        {
            up.segment(i * inputDim, inputDim) = u_data[u_data.size() - m_horizonLength + i];
            yp.segment(i * outputDim, outputDim) = y_data[y_data.size() - m_horizonLength + i];
        }

        // --- Build and solve the optimization problem ---

        // ------ OSQP Setup (Moved here) ------
        int n = Hu.cols();             // Number of variables (size of g)
        int m = up.rows() + yp.rows(); // Number of equality constraints
        // Cast the solver to the derived class OSQPSolver
        DataDrivenMPC::OSQPSolver *osqp_solver = dynamic_cast<DataDrivenMPC::OSQPSolver *>(m_solver);
        if (osqp_solver == nullptr)
        {
            throw std::runtime_error("Error: Solver provided is not an OSQPSolver instance");
        }
        // OSQP setup (moved from OSQPSolver)
        OSQPData *data = (OSQPData *)malloc(sizeof(OSQPData));
        if (!data)
        {
            throw std::runtime_error("Failed to allocate OSQPData.");
        }
        data->n = n;
        data->m = m;
        data->P = (csc *)malloc(sizeof(csc));
        data->A = (csc *)malloc(sizeof(csc));
        data->q = (double *)malloc(sizeof(double) * n);
        data->l = (double *)malloc(sizeof(double) * m);
        data->u = (double *)malloc(sizeof(double) * m);

        if (!data->P || !data->A || !data->q || !data->l || !data->u)
        {
            free(data->P);
            free(data->q);
            free(data->A);
            free(data->l);
            free(data->u);
            free(data);
            throw std::runtime_error("Failed to allocate memory for OSQP data.");
        }
        data->P->m = n;
        data->P->n = n;
        data->P->nzmax = n * n;
        data->P->x = (double *)malloc(sizeof(double) * data->P->nzmax);
        data->P->i = (c_int *)malloc(sizeof(c_int) * data->P->nzmax);
        data->P->p = (c_int *)malloc(sizeof(c_int) * (n + 1));

        if (!data->P->x || !data->P->i || !data->P->p)
        {
            free(data->P->x);
            free(data->P->i);
            free(data->P->p);
            free(data->P);
            free(data->q);
            free(data->A);
            free(data->l);
            free(data->u);
            free(data);

            throw std::runtime_error("Failed to allocate memory for OSQP P data.");
        }

        data->A->m = m;
        data->A->n = n;
        data->A->nzmax = n * m;
        data->A->x = (double *)malloc(sizeof(double) * data->A->nzmax);
        data->A->i = (c_int *)malloc(sizeof(c_int) * data->A->nzmax);
        data->A->p = (c_int *)malloc(sizeof(c_int) * (n + 1));

        if (!data->A->x || !data->A->i || !data->A->p)
        {
            free(data->P->x);
            free(data->P->i);
            free(data->P->p);
            free(data->P);

            free(data->q);

            free(data->A->x);
            free(data->A->i);
            free(data->A->p);
            free(data->A);

            free(data->l);
            free(data->u);
            free(data);
            throw std::runtime_error("Failed to allocate memory for OSQP A data.");
        }
        OSQPSettings *settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
        if (settings)
        {
            osqp_set_default_settings(settings);
            settings->verbose = 0; // Disable verbose output
        }
        else
        {
            free(data->P->x);
            free(data->P->i);
            free(data->P->p);
            free(data->P);
            free(data->q);
            free(data->A->x);
            free(data->A->i);
            free(data->A->p);
            free(data->A);
            free(data->l);
            free(data->u);
            free(data);
            throw std::runtime_error("Failed to allocate memory for settings.");
        }

        // Setup workspace (after allocating ALL data)
        OSQPWorkspace *work = nullptr; // Temporary workspace pointer
        if (::osqp_setup(&work, data, settings) != 0)
        { // Use ::osqp_setup
            free(settings);
            // Free data if setup fails.  Use standard free, not freeProblemData.
            free(data->P->x);
            free(data->P->i);
            free(data->P->p);
            free(data->P);
            free(data->q);
            free(data->A->x);
            free(data->A->i);
            free(data->A->p);
            free(data->A);
            free(data->l);
            free(data->u);
            free(data);
            throw std::runtime_error("Failed to set up OSQP workspace.");
        }
        free(settings); // Settings can be freed after the setup
        // Store the workspace in the solver. From now, the solver is responsible for the workspace
        osqp_solver->setWorkspace(work); // Use dynamic_cast to access setWorkspace
        osqp_solver->setData(data);      // Use dynamic_cast to access setData

        // -----------

        // 1.  Declare the optimization variable g
        Eigen::VectorXd g_optimal;

        // 2.  Build the optimization problem (set up matrices and vectors for the QP solver)
        buildOptimizationProblem(Hu.getMatrix(), Hy.getMatrix(), up, yp, reference, u_prev, g_optimal);

        // 3. Extract the optimal control input (first element of u_f)
        Eigen::VectorXd u_optimal(inputDim);
        // Optimal control input is obtained from the first `inputDim` elements of `Hu * g_optimal`
        u_optimal = (Hu.getMatrix() * g_optimal).head(inputDim);

        return u_optimal;
    }

    void DDMPC::buildOptimizationProblem(const Eigen::MatrixXd &Hu, const Eigen::MatrixXd &Hy,
                                         const Eigen::VectorXd &up, const Eigen::VectorXd &yp,
                                         const Eigen::VectorXd &reference, const Eigen::VectorXd &u_prev,
                                         Eigen::VectorXd &g_optimal)
    {

        // --- Build the matrices for the QP problem ---

        // Get dimensions
        int inputDim = up.rows() / m_horizonLength;
        int outputDim = yp.rows() / m_horizonLength;
        int n_g = Hu.cols(); // Number of columns in Hankel matrices (size of g)

        // --- Cost Function ---

        // P (Hessian)
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n_g, n_g);

        // Create block diagonal Q_block and R_block
        // Q_block now uses m_horizonLength (L), NOT m_predictionHorizon (N)
        Eigen::MatrixXd Q_block = Eigen::MatrixXd::Zero(m_horizonLength * outputDim, m_horizonLength * outputDim);
        for (int i = 0; i < m_horizonLength; ++i)
        {
            Q_block.block(i * outputDim, i * outputDim, outputDim, outputDim) = m_Q;
        }
        Eigen::MatrixXd R_block = Eigen::MatrixXd::Zero(m_controlHorizon * inputDim, m_controlHorizon * inputDim);
        for (int i = 0; i < m_controlHorizon; ++i)
        {
            R_block.block(i * inputDim, i * inputDim, inputDim, inputDim) = m_R;
        }

        // Compute H_u_delta
        Eigen::MatrixXd Hu_delta(m_controlHorizon * inputDim, n_g);
        // We have [Hu(1,:); Hu(2,:)-Hu(1,:), ..., Hu(M,:)-Hu(M-1,:)] * g
        // First block is just the first input
        Hu_delta.block(0, 0, inputDim, n_g) = Hu.block(0, 0, inputDim, n_g);
        // Other blocks are differences of control inputs
        for (int i = 1; i < m_controlHorizon; i++)
        {
            Hu_delta.block(i * inputDim, 0, inputDim, n_g) =
                Hu.block(i * inputDim, 0, inputDim, n_g) - Hu.block((i - 1) * inputDim, 0, inputDim, n_g);
        }

        // std::cout << "R_block dimensions: " << R_block.rows() << " x " << R_block.cols() << std::endl;
        P = Hy.transpose() * Q_block * Hy + Hu_delta.transpose() * R_block * Hu_delta;
        P = 2 * P; // Multiply by 2 because the QP solver expects 0.5*x'Px + q'x

        // q (Gradient)
        // take the first L elements of the reference
        Eigen::VectorXd q = -2 * Hy.transpose() * Q_block * reference.head(m_horizonLength * outputDim);

        // --- Constraints ---

        // Equality constraints:  Ag = b
        // [ Hu ] [ g ] = [ up ]
        // [ Hy ]       = [ yp ]

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(up.rows() + yp.rows(), n_g);
        A.topRows(up.rows()) = Hu;
        A.bottomRows(yp.rows()) = Hy;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(up.rows() + yp.rows());
        b.head(up.rows()) = up;
        b.tail(yp.rows()) = yp;

        // Inequality constraints:  l <= Cx <= u
        // We have several optional constraints:
        // Input constraints
        // Output constraints
        // Delta Input constraints
        std::vector<Eigen::MatrixXd> C_vec;
        std::vector<Eigen::VectorXd> l_vec;
        std::vector<Eigen::VectorXd> u_vec;

        if (m_useInputConstraints)
        {
            // u_min <= Hu * g <= u_max   for the first M blocks (control horizon)
            Eigen::MatrixXd C_u = Eigen::MatrixXd::Zero(m_controlHorizon * inputDim, n_g);
            for (int i = 0; i < m_controlHorizon; ++i)
            {
                C_u.block(i * inputDim, 0, inputDim, n_g) = Hu.block(i * inputDim, 0, inputDim, n_g);
            }
            C_vec.push_back(C_u);
            l_vec.push_back(m_uMin.replicate(m_controlHorizon, 1)); // Stack u_min M times
            u_vec.push_back(m_uMax.replicate(m_controlHorizon, 1)); // Stack u_max M times
        }

        if (m_useOutputConstraints)
        {
            // y_min <= Hy * g <= y_max  for the first N blocks (prediction horizon)
            Eigen::MatrixXd C_y = Eigen::MatrixXd::Zero(m_predictionHorizon * outputDim, n_g);
            for (int i = 0; i < m_predictionHorizon; i++)
            {
                C_y.block(i * outputDim, 0, outputDim, n_g) = Hy.block(i * outputDim, 0, outputDim, n_g);
            }
            C_vec.push_back(C_y);
            l_vec.push_back(m_yMin.replicate(m_predictionHorizon, 1)); // Stack y_min N times
            u_vec.push_back(m_yMax.replicate(m_predictionHorizon, 1)); // Stack y_max N times
        }

        if (m_useDeltaInputConstraints)
        {
            // delta_u_min <= delta_u <= delta_u_max
            //  delta_u = [ u(k) - u(k-1); u(k+1) - u(k); ...; u(k+M-1) - u(k+M-2) ]
            //            = [ Hu(1,:)*g - u(k-1);  Hu(2,:)*g - Hu(1,:)*g; ...; Hu(M,:)*g - Hu(M-1,:)*g ]
            Eigen::MatrixXd C_delta_u = Eigen::MatrixXd::Zero(m_controlHorizon * inputDim, n_g);

            // First block:  u(k) - u(k-1)
            C_delta_u.block(0, 0, inputDim, n_g) = Hu.block(0, 0, inputDim, n_g);
            Eigen::VectorXd l_delta_u_first_block = m_deltaUmin;
            Eigen::VectorXd u_delta_u_first_block = m_deltaUmax;
            for (int i = 0; i < inputDim; ++i)
            {
                l_delta_u_first_block(i) += -u_prev(i); //  -u(k-1)
                u_delta_u_first_block(i) += -u_prev(i); //  -u(k-1)
            }

            // Remaining blocks: u(k+i) - u(k+i-1)
            for (int i = 1; i < m_controlHorizon; ++i)
            {
                C_delta_u.block(i * inputDim, 0, inputDim, n_g) = Hu.block(i * inputDim, 0, inputDim, n_g) - Hu.block((i - 1) * inputDim, 0, inputDim, n_g);
            }

            C_vec.push_back(C_delta_u);
            // adjust and stack
            Eigen::VectorXd l_delta_u(m_controlHorizon * inputDim);
            Eigen::VectorXd u_delta_u(m_controlHorizon * inputDim);
            l_delta_u.head(inputDim) = l_delta_u_first_block;
            u_delta_u.head(inputDim) = u_delta_u_first_block;
            l_delta_u.tail((m_controlHorizon - 1) * inputDim) = m_deltaUmin.replicate(m_controlHorizon - 1, 1); // Stack delta_u_min (M-1) times
            u_delta_u.tail((m_controlHorizon - 1) * inputDim) = m_deltaUmax.replicate(m_controlHorizon - 1, 1); // Stack delta_u_max (M-1) times
            l_vec.push_back(l_delta_u);
            u_vec.push_back(u_delta_u);
        }

        // Concatenate all inequality constraints
        Eigen::MatrixXd C;
        Eigen::VectorXd l, u;

        if (C_vec.size() > 0) // Only if there is at least one constraint
        {
            int totalRows = 0;
            for (const auto &matrix : C_vec)
            {
                totalRows += matrix.rows();
            }
            C.resize(totalRows, n_g);
            l.resize(totalRows);
            u.resize(totalRows);

            int currentRow = 0;
            for (size_t i = 0; i < C_vec.size(); ++i)
            {
                C.block(currentRow, 0, C_vec[i].rows(), n_g) = C_vec[i];
                l.segment(currentRow, l_vec[i].size()) = l_vec[i];
                u.segment(currentRow, u_vec[i].size()) = u_vec[i];
                currentRow += C_vec[i].rows();
            }
            // m_solver->setInequalityConstraints(C, l, u);
        }
        else // No inequality constraints
        {
            std::cout << "No inequality constraints! is this correct ?" << std::endl;
            int eq_constraints_size = A.rows();

            C.resize(0, n_g);
            l.resize(0);
            u.resize(0);
            // m_solver->setInequalityConstraints(C, l, u);
        }

        // --- Call the QP solver ---
        m_solver->setObjective(P, q);
        m_solver->setEqualityConstraints(A, b);
        if (C_vec.size() > 0)
        { // Only call if there are inequality constraints
            m_solver->setInequalityConstraints(C, l, u);
        }

        bool solverSuccess = m_solver->solve(g_optimal); // Solve the QP

        if (!solverSuccess)
        {
            throw std::runtime_error("QP solver failed.");
        }
    }

    void DDMPC::checkInputData(const std::vector<Eigen::VectorXd> &u_data, const std::vector<Eigen::VectorXd> &y_data) const
    {
        if (u_data.size() < static_cast<size_t>(m_horizonLength))
        {
            throw std::invalid_argument("Insufficient input data for the given horizon length. Expected at least " +
                                        std::to_string(m_horizonLength) + " samples, but got " +
                                        std::to_string(u_data.size()) + ".");
        }

        if (y_data.size() < static_cast<size_t>(m_horizonLength))
        {
            throw std::invalid_argument("Insufficient output data for the given horizon length.  Expected at least " +
                                        std::to_string(m_horizonLength) + " samples, but got " +
                                        std::to_string(y_data.size()) + ".");
        }

        // Check if dimensions are consistent
        int inputDim = u_data[0].rows();
        int outputDim = y_data[0].rows();
        for (size_t i = 1; i < u_data.size(); ++i)
        {
            if (u_data[i].rows() != inputDim)
            {
                throw std::invalid_argument("Inconsistent input vector dimensions across the data sequence.");
            }
        }
        for (size_t i = 1; i < y_data.size(); ++i)
        {
            if (y_data[i].rows() != outputDim)
            {
                throw std::invalid_argument("Inconsistent output vector dimensions across the data sequence.");
            }
        }
    }

} // namespace DataDrivenMPC