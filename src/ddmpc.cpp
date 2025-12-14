#include "data-driven-mpc/data-driven-mpc.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>
#include <chrono>
#include <Eigen/Sparse>

namespace DataDrivenMPC
{

    DDMPC::DDMPC(int Tini, int predictionHorizon, int controlHorizon,
                 const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, std::unique_ptr<QPSolver> solver,
                 std::string solver_name, double lambda_u, double lambda_y, double lambda_g,
                 double scale_g, double scale_u, double scale_y)
        : m_Tini(Tini), m_predictionHorizon(predictionHorizon),
          m_controlHorizon(controlHorizon), m_Q(Q), m_R(R), m_solver(std::move(solver)),
          m_solver_name(solver_name), m_lambda_u(lambda_u), m_lambda_y(lambda_y), m_lambda_g(lambda_g),
          m_u_dim(R.rows()), m_y_dim(Q.rows()), m_num_g(0),
          m_n_variables(0), m_n_constraints(0), m_is_problem_setup(false),
          m_is_solver_initialized(false), m_num_t_g(0), m_num_t_y(0),
          m_scale_g(scale_g), m_scale_u(scale_u), m_scale_y(scale_y),
          m_useInputConstraints(false), m_useOutputConstraints(false), m_useDeltaInputConstraints(false),
          m_data_requires_pe_check(true)
    {
        if (m_controlHorizon > m_predictionHorizon)
        {
            throw std::invalid_argument("Control horizon cannot be greater than prediction horizon.");
        }
        if (m_Tini <= 0 || m_predictionHorizon <= 0 || m_controlHorizon <= 0)
        {
            throw std::invalid_argument("Horizons must be positive integers.");
        }
        if (Q.rows() != Q.cols() || R.rows() != R.cols() || Q.rows() != m_y_dim || R.rows() != m_u_dim)
        {
            throw std::invalid_argument("Q and R matrices must be square and have compatible dimensions.");
        }
        if (lambda_u < 0 || lambda_y < 0 || lambda_g < 0)
        {
            throw std::invalid_argument("Regularization parameters must be non-negative.");
        }
        if (scale_g <= 0.0 || scale_u <= 0.0 || scale_y <= 0.0)
        {
            throw std::invalid_argument("Scaling factors (scale_g, scale_u, scale_y) must be positive.");
        }

        bool ok = true;
        ok = ok & m_solver->instantiateSolver(m_solver_name);

        // --- Set Common Defaults ---
        double common_eps_abs = 1e-4;
        double common_eps_rel = 1e-4;
        int common_max_iter = 4000;
        bool common_verbose = false;

        ok = ok & m_solver->setRealNumberParameter("eps_abs", common_eps_abs);
        ok = ok & m_solver->setRealNumberParameter("eps_rel", common_eps_rel);
        ok = ok & m_solver->setIntegerParameter("max_iter", common_max_iter);
        ok = ok & m_solver->setBooleanParameter("verbose", common_verbose);

        // --- Apply Solver-Specific Settings/Overrides ---
        if (m_solver_name == "osqp")
        {
            ok = ok & m_solver->setBooleanParameter("polish", false);
            ok = ok & m_solver->setBooleanParameter("adaptive_rho", true);
            ok = ok & m_solver->setRealNumberParameter("sigma", 1e-6);
            ok = ok & m_solver->setBooleanParameter("warm_starting", true);

        }
        else if (m_solver_name == "proxqp")
        {
            ok = ok & m_solver->setBooleanParameter("check_duality_gap", true);
            // ok = ok & m_solver->setRealNumberParameter("default_mu_in", 1e-4);
            // ok = ok & m_solver->setIntegerParameter("nb_iterative_refinement", 0);
            ok = ok & m_solver->setStringParameter("initial_guess", "WARM_START_WITH_PREVIOUS_RESULT");
        }
        else
        {
            throw std::invalid_argument("Unsupported solver name.");
        }
        // Enable solver scaling? OSQP has built-in scaling.
        // This may conflict with the scaling matrix S.
        if (!ok)
        {
            throw std::runtime_error("Failed to instantiate or configure QP solver.");
        }
        // Initialize m_Q_vec and m_R_vec with the initial Q and R matrices
        m_Q_vec.resize(m_predictionHorizon, m_Q);
        m_R_vec.resize(m_predictionHorizon, m_R);
    }

    void DDMPC::updateRegularizationWeights(double lambda_u, double lambda_y, double lambda_g)
    {
        setLambdaU(lambda_u);
        setLambdaY(lambda_y);
        setLambdaG(lambda_g);
        m_is_problem_setup = false;
        m_is_solver_initialized = false;
    }

    void DDMPC::setLambdaU(double lambda_u)
    {
        if (lambda_u < 0)
            throw std::invalid_argument("lambda_u must be non-negative.");
        if (m_lambda_u != lambda_u)
        {
            m_lambda_u = lambda_u;
            if (m_is_problem_setup)
                updateObjective(Eigen::VectorXd());
        }
    }

    void DDMPC::setLambdaY(double lambda_y)
    {
        if (lambda_y < 0)
            throw std::invalid_argument("lambda_y must be non-negative.");
        bool requires_rebuild = (m_lambda_y <= 0 && lambda_y > 0) || (m_lambda_y > 0 && lambda_y <= 0);
        m_lambda_y = lambda_y;
        if (requires_rebuild && m_is_problem_setup)
        {
            m_is_problem_setup = false; // Structure changes
            m_is_solver_initialized = false;
        }
        else if (m_is_problem_setup)
        {
            updateObjective(Eigen::VectorXd()); // Only objective changes
        }
    }

    void DDMPC::setLambdaG(double lambda_g)
    {
        if (lambda_g < 0)
            throw std::invalid_argument("lambda_g must be non-negative.");
        bool requires_rebuild = (m_lambda_g <= 0 && lambda_g > 0) || (m_lambda_g > 0 && lambda_g <= 0);
        m_lambda_g = lambda_g;
        if (requires_rebuild && m_is_problem_setup)
        {
            m_is_problem_setup = false; // Structure changes
            m_is_solver_initialized = false;
        }
        else if (m_is_problem_setup)
        {
            updateObjective(Eigen::VectorXd()); // Only objective changes
        }
    }
    double DDMPC::getLambdaU() const { return m_lambda_u; }
    double DDMPC::getLambdaY() const { return m_lambda_y; }
    double DDMPC::getLambdaG() const { return m_lambda_g; }

    // --- Constraint setters (may require QP rebuild ---
    void DDMPC::setInputConstraints(const Eigen::VectorXd &u_min, const Eigen::VectorXd &u_max)
    {
        if (u_min.size() != u_max.size() || u_min.size() != m_u_dim)
            throw std::invalid_argument("u_min/max size mismatch.");
        m_uMin = u_min;
        m_uMax = u_max;
        bool structure_changed = !m_useInputConstraints;
        m_useInputConstraints = true;
        if (structure_changed && m_is_problem_setup)
        {
            m_is_problem_setup = false;
            m_is_solver_initialized = false;
        }
        else if (m_is_problem_setup)
        {
            // Only bounds change, update later in dynamic constraints phase
            // m_lower_bound.segment(m_n_variables - m_u_dim, m_u_dim) = m_uMin;
            // m_upper_bound.segment(m_n_variables - m_u_dim, m_u_dim) = m_uMax;
        }
    }
    void DDMPC::setOutputConstraints(const Eigen::VectorXd &y_min, const Eigen::VectorXd &y_max)
    {
        if (y_min.size() != y_max.size() || y_min.size() != m_y_dim)
            throw std::invalid_argument("y_min/max size mismatch.");
        m_yMin = y_min;
        m_yMax = y_max;
        bool structure_changed = !m_useOutputConstraints;
        m_useOutputConstraints = true;
        if (structure_changed && m_is_problem_setup)
        {
            m_is_problem_setup = false;
            m_is_solver_initialized = false;
        }
        else if (m_is_problem_setup)
        {
            // Only bounds change, update later in dynamic constraints phase
            // m_lower_bound.segment(m_n_variables - m_y_dim, m_y_dim) = m_yMin;
            // m_upper_bound.segment(m_n_variables - m_y_dim, m_y_dim) = m_yMax;
        }
    }
    void DDMPC::setDeltaInputConstraints(const Eigen::VectorXd &delta_u_min, const Eigen::VectorXd &delta_u_max)
    {
        if (delta_u_min.size() != delta_u_max.size() || delta_u_min.size() != m_u_dim)
            throw std::invalid_argument("delta_u size mismatch.");
        m_deltaUmin = delta_u_min;
        m_deltaUmax = delta_u_max;
        bool structure_changed = !m_useDeltaInputConstraints;
        m_useDeltaInputConstraints = true;
        if (structure_changed && m_is_problem_setup)
        {
            m_is_problem_setup = false;
            m_is_solver_initialized = false;
        }
        else if (m_is_problem_setup)
        {
            // Only bounds change, update later in dynamic constraints phase
        }
    }

    // --- Weight updates only affect objective, handled by updateObjective ---
    void DDMPC::updateWeights(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
    {
        if (Q.rows() != Q.cols() || Q.rows() != m_y_dim)
            throw std::invalid_argument("Q matrix dimensions.");
        if (R.rows() != R.cols() || R.rows() != m_u_dim)
            throw std::invalid_argument("R matrix dimensions.");
        for (int i = 0; i < m_predictionHorizon; ++i)
        {
            m_Q_vec[i] = Q;
            m_R_vec[i] = R;
        }
        if (m_is_problem_setup)
            updateObjective(Eigen::VectorXd());
    }
    void DDMPC::updateWeights(const std::vector<Eigen::MatrixXd> &Q_vec, const std::vector<Eigen::MatrixXd> &R_vec)
    {
        if (Q_vec.size() != m_predictionHorizon || R_vec.size() != m_predictionHorizon)
            throw std::invalid_argument("Q/R_vec size mismatch.");
        // Dimension checks...
        m_Q_vec = Q_vec;
        m_R_vec = R_vec;
        if (m_is_problem_setup)
            updateObjective(Eigen::VectorXd());
    }

    // NORMALIZATION: Helper to build the diagonal scaling matrix S
    Eigen::SparseMatrix<double> DDMPC::buildScalingMatrix() const
    {

        // --- Use Dimensions Consistent with setupProblem ---
        const int g_dim = m_num_g;
        const int u_dim_pred = m_predictionHorizon * m_u_dim;
        const int y_dim_pred = m_predictionHorizon * m_y_dim;
        const int s_u_dim = m_Tini * m_u_dim;
        const int s_y_dim = m_Tini * m_y_dim;

        const int t_g_dim = m_num_t_g;
        const int t_y_dim = m_num_t_y;

        // --- Define Offsets ---
        const int g_offset = 0;
        const int u_offset = g_offset + g_dim;
        const int y_offset = u_offset + u_dim_pred;
        const int s_u_offset = y_offset + y_dim_pred;
        const int s_y_offset = s_u_offset + s_u_dim;
        const int t_g_offset = s_y_offset + s_y_dim;
        const int t_y_offset = t_g_offset + t_g_dim;

        // --- Calculate Total Variables (as a cross-check) ---
        const int n_vars_check = g_dim + u_dim_pred + y_dim_pred + s_u_dim + s_y_dim + t_g_dim + t_y_dim;
        if (n_vars_check != m_n_variables)
        {
            throw std::logic_error("Variable count mismatch between setupProblem and buildScalingMatrix!");
        }

        // --- Build the Diagonal Scaling Matrix S ---
        Eigen::SparseMatrix<double> S(m_n_variables, m_n_variables);
        std::vector<Eigen::Triplet<double>> s_triplets;
        s_triplets.reserve(m_n_variables); // Reserve space for diagonal entries

        // Scale g block by m_scale_g
        for (int i = 0; i < g_dim; ++i)
            s_triplets.emplace_back(g_offset + i, g_offset + i, m_scale_g);

        // Scale u block by m_scale_u
        for (int i = 0; i < u_dim_pred; ++i)
            s_triplets.emplace_back(u_offset + i, u_offset + i, m_scale_u);

        // Scale y block by m_scale_y
        for (int i = 0; i < y_dim_pred; ++i)
            s_triplets.emplace_back(y_offset + i, y_offset + i, m_scale_y);

        // Scale s_u block like u (by m_scale_u)
        for (int i = 0; i < s_u_dim; ++i)
            s_triplets.emplace_back(s_u_offset + i, s_u_offset + i, m_scale_u);

        // Scale s_y block like y (by m_scale_y)
        for (int i = 0; i < s_y_dim; ++i)
            s_triplets.emplace_back(s_y_offset + i, s_y_offset + i, m_scale_y);

        // Scale t_g block like g (by m_scale_g), if it exists
        if (t_g_dim > 0)
        {
            for (int i = 0; i < t_g_dim; ++i)
                s_triplets.emplace_back(t_g_offset + i, t_g_offset + i, m_scale_g);
        }

        // Scale t_y block like s_y (-> like y, by m_scale_y), if it exists
        if (t_y_dim > 0)
        {
            for (int i = 0; i < t_y_dim; ++i)
                s_triplets.emplace_back(t_y_offset + i, t_y_offset + i, m_scale_y);
        }

        // Final check on number of triplets added
        if (s_triplets.size() != static_cast<size_t>(m_n_variables))
        {
            throw std::logic_error("Incorrect number of diagonal elements added to scaling matrix S!");
        }

        S.setFromTriplets(s_triplets.begin(), s_triplets.end());
        return S;
    }

    Eigen::VectorXd DDMPC::solve(const std::vector<Eigen::VectorXd> &u_data,
                                 const std::vector<Eigen::VectorXd> &y_data,
                                 const Eigen::VectorXd &reference,
                                 const Eigen::VectorXd &u_prev,
                                 const std::vector<Eigen::VectorXd> &u_data_ini,
                                 const std::vector<Eigen::VectorXd> &y_data_ini)
    {
        auto start = std::chrono::high_resolution_clock::now();

        checkInputData(u_data, y_data);
        if (reference.size() != 0 && reference.size() != m_y_dim)
        { // Allow empty reference
            throw std::invalid_argument("Reference vector has incorrect dimensions.");
        }
        if (static_cast<int>(u_data_ini.size()) != m_Tini || static_cast<int>(y_data_ini.size()) != m_Tini)
        {
            throw std::invalid_argument("Initial data vectors size mismatch.");
        }

        m_initial_u_data = u_data;
        m_initial_y_data = y_data;

        // TODO: handle better data changes
        bool data_changed = (u_data != m_initial_u_data) || (y_data != m_initial_y_data);
        if (data_changed && m_is_problem_setup) {
            std::cout << "Warning: Input data changed but problem was already set up. Rebuilding." << std::endl;
            m_is_problem_setup = false; // Force rebuild if data changed
             // m_data_requires_pe_check = true; // setupProblem will set this anyway
        }

        // --- Problem Setup (if needed) ---
        if (!m_is_problem_setup)
        {
            setupProblem(u_data, y_data);
        }

        // --- Prepare Dynamic Data ---
        Eigen::VectorXd u_ini(m_Tini * m_u_dim);
        Eigen::VectorXd y_ini(m_Tini * m_y_dim);
        for (int i = 0; i < m_Tini; ++i)
        {
            u_ini.segment(i * m_u_dim, m_u_dim) = u_data_ini[i];
            y_ini.segment(i * m_y_dim, m_y_dim) = y_data_ini[i];
        }

        // --- Update Dynamic Parts of QP ---
        updateDynamicConstraints(u_ini, y_ini, reference, u_prev);
        updateObjective(reference);

        // --- Initialize or Update Solver ---
        if (!m_is_solver_initialized)
        {
            m_solver->setNumberOfVariables(m_n_variables);
            m_solver->setNumberOfConstraints(m_n_constraints);
            m_solver->setHessianMatrix(m_H_scaled);
            m_solver->setGradient(m_f_scaled);
            m_solver->setLinearConstraintsMatrix(m_A_scaled);
            m_solver->setLowerBound(m_lower_bound);
            m_solver->setUpperBound(m_upper_bound);

            if (!m_solver->initSolver())
            {
                throw std::runtime_error("Failed to initialize QP solver.");
            }
            m_is_solver_initialized = true;
        }
        else
        {
            m_solver->updateGradient(m_f_scaled);
            m_solver->updateBounds(m_lower_bound, m_upper_bound);
            // Potentially update A_scaled if Hankel matrices change (not supported here yet)
            // m_solver->updateLinearConstraintsMatrix(m_A_scaled);
            // Potentially update H_scaled if Q/R change
            // m_solver->updateHessianMatrix(m_H_scaled); // Already done in updateObjective if needed
        }

        // --- Solve ---
        QpSolversEigen::ErrorExitFlag solver_status = m_solver->solveProblem();
        if (solver_status != QpSolversEigen::ErrorExitFlag::NoError)
        {
            std::cerr << "Warning: QP solver finished with status: " << static_cast<int>(solver_status) << std::endl;
            // TODO: Consider more robust error handling (e.g., returning previous solution)
        }

        // --- Extract and Rescale Solution ---
        Eigen::VectorXd solution_scaled = m_solver->getSolution();
        Eigen::VectorXd u_optimal_physical(m_controlHorizon * m_u_dim);

        const int g_dim = m_num_g;
        const int u_offset = g_dim;

        if (solution_scaled.size() != m_n_variables)
        {
            std::cerr << "Error: Solver solution size mismatch! Expected " << m_n_variables << ", Got " << solution_scaled.size() << std::endl;
            // TODO: Handle error: return zero vector or previous solution?
            return Eigen::VectorXd::Zero(m_u_dim);
        }

        for (int i = 0; i < m_controlHorizon; ++i)
        {
            // Segment of scaled solution corresponding to u_{k+i}
            Eigen::VectorXd u_scaled_i = solution_scaled.segment(u_offset + i * m_u_dim, m_u_dim);
            // Rescale to physical units
            u_optimal_physical.segment(i * m_u_dim, m_u_dim) = u_scaled_i * m_scale_u;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cerr << "Elapsed time: " << elapsed.count() * 1000.0 << " ms" << std::endl;

        // Return the first optimal input in physical units

        return u_optimal_physical.head(m_u_dim);
    }

    void DDMPC::checkInputData(const std::vector<Eigen::VectorXd> &u_data, const std::vector<Eigen::VectorXd> &y_data) const {
        if (u_data.empty() || y_data.empty())
            throw std::invalid_argument("Input or output data vectors are empty.");
        if (u_data.size() != y_data.size())
            throw std::invalid_argument("Input and output data vectors must have the same length.");
        if (u_data[0].size() != m_u_dim)
            throw std::invalid_argument("Input data vector dimension mismatch.");
        if (y_data[0].size() != m_y_dim)
            throw std::invalid_argument("Output data vector dimension mismatch.");
     }
    void DDMPC::checkConstraintDimensions() const {
        if (m_useInputConstraints && m_uMin.size() != m_uMax.size())
            throw std::invalid_argument("Input constraints size mismatch.");
        if (m_useOutputConstraints && m_yMin.size() != m_yMax.size())
            throw std::invalid_argument("Output constraints size mismatch.");
        if (m_useDeltaInputConstraints && m_deltaUmin.size() != m_deltaUmax.size())
            throw std::invalid_argument("Delta input constraints size mismatch.");
    }

    void DDMPC::setupProblem(const std::vector<Eigen::VectorXd> &u_data, const std::vector<Eigen::VectorXd> &y_data)
    {
        HankelMatrix Hu(u_data, m_Tini + m_predictionHorizon);
        HankelMatrix Hy(y_data, m_Tini + m_predictionHorizon);
        m_num_g = Hu.cols();

        double pe_tolerance = 1e-6; // TODO: Make this configurable
        if (!Hu.isPersistentlyExciting(Hu.rows(), pe_tolerance) ||
            !Hy.isPersistentlyExciting(Hy.rows(), pe_tolerance))
        {
            std::cerr << "WARNING: Input/Output data might not be persistently exciting!" << std::endl;
            // TODO: Consider if this should be a fatal error or just a warning
            throw std::runtime_error("Input/Output data not persistently exciting.");
        }
        std::cout << "PE Check Passed in setupProblem." << std::endl;
        m_data_requires_pe_check = false;

        // --- Define dimensions ---
        const int g_dim = m_num_g;
        const int u_dim_pred = m_predictionHorizon * m_u_dim;
        const int y_dim_pred = m_predictionHorizon * m_y_dim;
        const int s_u_dim = m_Tini * m_u_dim;
        const int s_y_dim = m_Tini * m_y_dim;

        // Use consistent tolerance for checking if L1 terms are active
        m_num_t_g = (m_lambda_g > 1e-9) ? g_dim : 0;
        m_num_t_y = (m_lambda_y > 1e-9) ? s_y_dim : 0;
        const int t_g_dim = m_num_t_g;
        const int t_y_dim = m_num_t_y;

        // --- Define offsets ---
        const int g_offset = 0;
        const int u_offset = g_offset + g_dim;
        const int y_offset = u_offset + u_dim_pred;
        const int s_u_offset = y_offset + y_dim_pred;
        const int s_y_offset = s_u_offset + s_u_dim;
        const int t_g_offset = s_y_offset + s_y_dim;
        const int t_y_offset = t_g_offset + t_g_dim;

        m_n_variables = g_dim + u_dim_pred + y_dim_pred + s_u_dim + s_y_dim + t_g_dim + t_y_dim;

        // --- Hankel Blocks ---
        Eigen::MatrixXd Up = Hu.getMatrix().block(0, 0, m_Tini * m_u_dim, m_num_g);
        Eigen::MatrixXd Yp = Hy.getMatrix().block(0, 0, m_Tini * m_y_dim, m_num_g);
        Eigen::MatrixXd Uf = Hu.getMatrix().block(m_Tini * m_u_dim, 0, u_dim_pred, m_num_g);
        Eigen::MatrixXd Yf = Hy.getMatrix().block(m_Tini * m_y_dim, 0, y_dim_pred, m_num_g);

        // --- Build Original (Unscaled) QP Matrices FIRST ---
        Eigen::SparseMatrix<double> H_original(m_n_variables, m_n_variables);
        Eigen::VectorXd f_original = Eigen::VectorXd::Zero(m_n_variables);
        Eigen::SparseMatrix<double> A_original;
        // Bounds are initialized later

        // --- Calculate Constraint Dimensions (Mirroring Unscaled Version) ---
        const int core_eq_rows = (m_Tini * (m_u_dim + m_y_dim)) + (m_predictionHorizon * (m_u_dim + m_y_dim));
        const int input_ineq_rows = m_useInputConstraints ? m_controlHorizon * m_u_dim : 0;
        const int output_ineq_rows = m_useOutputConstraints ? m_predictionHorizon * m_y_dim : 0;
        const int delta_input_ineq_rows = m_useDeltaInputConstraints ? m_controlHorizon * m_u_dim : 0;
        // L1 constraints use the Ax >= l formulation (2 rows per variable)
        const int l1_g_ineq_rows = (t_g_dim > 0) ? 2 * g_dim : 0;
        const int l1_y_ineq_rows = (t_y_dim > 0) ? 2 * s_y_dim : 0;
        const int total_ineq_rows = input_ineq_rows + output_ineq_rows + delta_input_ineq_rows + l1_g_ineq_rows + l1_y_ineq_rows;
        m_n_constraints = core_eq_rows + total_ineq_rows;

        // --- Hessian (Original H) ---
        std::vector<Eigen::Triplet<double>> H_triplets;
        H_triplets.reserve(m_predictionHorizon * (m_u_dim * m_u_dim + m_y_dim * m_y_dim) + s_u_dim); // Estimate
        // R cost (u)
        for (int i = 0; i < m_predictionHorizon; ++i)
        {
            for (int r = 0; r < m_u_dim; ++r)
            {
                for (int c = 0; c < m_u_dim; ++c)
                {
                    if (std::abs(m_R_vec[i](r, c)) > 1e-15) // Tolerance for zero
                        H_triplets.emplace_back(u_offset + i * m_u_dim + r, u_offset + i * m_u_dim + c, m_R_vec[i](r, c));
                }
            }
        }
        // Q cost (y)
        for (int i = 0; i < m_predictionHorizon; ++i)
        {
            for (int r = 0; r < m_y_dim; ++r)
            {
                for (int c = 0; c < m_y_dim; ++c)
                {
                    if (std::abs(m_Q_vec[i](r, c)) > 1e-15) // Tolerance for zero
                        H_triplets.emplace_back(y_offset + i * m_y_dim + r, y_offset + i * m_y_dim + c, m_Q_vec[i](r, c));
                }
            }
        }
        // lambda_u cost (s_u) - L2 Regularization
        if (m_lambda_u > 1e-9)
        {
            for (int i = 0; i < s_u_dim; ++i)
                H_triplets.emplace_back(s_u_offset + i, s_u_offset + i, m_lambda_u);
        }
        H_original.resize(m_n_variables, m_n_variables);
        H_original.setFromTriplets(H_triplets.begin(), H_triplets.end());

        // --- Gradient (Original f - Base) ---
        // Reference part set dynamically in updateObjective
        // L1 parts are constant linear terms
        f_original.setZero();
        if (t_g_dim > 0)
            f_original.segment(t_g_offset, t_g_dim).setConstant(m_lambda_g);
        if (t_y_dim > 0)
            f_original.segment(t_y_offset, t_y_dim).setConstant(m_lambda_y);
        // Store this base gradient for later updates
        m_f_original_base = f_original;

        // --- Constraint Matrix (Original A - Mirroring Unscaled) ---
        std::vector<Eigen::Triplet<double>> A_triplets;
        // Estimate non-zeros (rough)
        A_triplets.reserve(core_eq_rows * (g_dim + 1) + total_ineq_rows * 2);
        int current_row = 0;

        // Core Equality Constraints (Ax = b form -> lower=upper=b)
        // Up*g - s_u = u_ini
        for (int i = 0; i < m_Tini * m_u_dim; ++i)
        {
            for (int j = 0; j < g_dim; ++j)
                if (std::abs(Up(i, j)) > 1e-15)
                    A_triplets.emplace_back(current_row, g_offset + j, Up(i, j));
            A_triplets.emplace_back(current_row, s_u_offset + i, -1.0);
            current_row++;
        }
        // Yp*g - s_y = y_ini
        for (int i = 0; i < m_Tini * m_y_dim; ++i)
        {
            for (int j = 0; j < g_dim; ++j)
                if (std::abs(Yp(i, j)) > 1e-15)
                    A_triplets.emplace_back(current_row, g_offset + j, Yp(i, j));
            A_triplets.emplace_back(current_row, s_y_offset + i, -1.0);
            current_row++;
        }
        // Uf*g - u = 0
        for (int i = 0; i < u_dim_pred; ++i)
        {
            for (int j = 0; j < g_dim; ++j)
                if (std::abs(Uf(i, j)) > 1e-15)
                    A_triplets.emplace_back(current_row, g_offset + j, Uf(i, j));
            A_triplets.emplace_back(current_row, u_offset + i, -1.0);
            current_row++;
        }
        // Yf*g - y = 0
        for (int i = 0; i < y_dim_pred; ++i)
        {
            for (int j = 0; j < g_dim; ++j)
                if (std::abs(Yf(i, j)) > 1e-15)
                    A_triplets.emplace_back(current_row, g_offset + j, Yf(i, j));
            A_triplets.emplace_back(current_row, y_offset + i, -1.0);
            current_row++;
        }
        assert(current_row == core_eq_rows); // Sanity check

        // Inequality Constraints (l <= Ax <= u form)
        // Input constraints: u_min <= u_i <= u_max => u_min <= 1*u_i <= u_max
        if (m_useInputConstraints)
        {
            for (int i = 0; i < m_controlHorizon; ++i)
            {
                for (int j = 0; j < m_u_dim; ++j)
                {
                    A_triplets.emplace_back(current_row, u_offset + i * m_u_dim + j, 1.0);
                    current_row++;
                }
            }
        }
        // Output constraints: y_min <= y_i <= y_max => y_min <= 1*y_i <= y_max
        if (m_useOutputConstraints)
        {
            for (int i = 0; i < m_predictionHorizon; ++i)
            {
                for (int j = 0; j < m_y_dim; ++j)
                {
                    A_triplets.emplace_back(current_row, y_offset + i * m_y_dim + j, 1.0);
                    current_row++;
                }
            }
        }
        // Delta input constraints: delta_u_min <= u_i - u_{i-1} <= delta_u_max
        if (m_useDeltaInputConstraints)
        {
            for (int i = 0; i < m_controlHorizon; ++i)
            {
                for (int j = 0; j < m_u_dim; ++j)
                {
                    A_triplets.emplace_back(current_row, u_offset + i * m_u_dim + j, 1.0); // u_i
                    if (i > 0)
                        A_triplets.emplace_back(current_row, u_offset + (i - 1) * m_u_dim + j, -1.0); // -u_{i-1}
                    // u_prev handled by bounds in updateDynamicConstraints
                    current_row++;
                }
            }
        }
        // L1 constraints for g: t_g >= g and t_g >= -g (Ax >= l form)
        // -> -g + t_g >= 0
        // ->  g + t_g >= 0
        if (t_g_dim > 0)
        {
            for (int i = 0; i < g_dim; ++i)
            {
                // Row for: -g_i + t_g_i >= 0
                A_triplets.emplace_back(current_row, g_offset + i, -1.0);
                A_triplets.emplace_back(current_row, t_g_offset + i, 1.0);
                current_row++;
                // Row for: g_i + t_g_i >= 0
                A_triplets.emplace_back(current_row, g_offset + i, 1.0);
                A_triplets.emplace_back(current_row, t_g_offset + i, 1.0);
                current_row++;
            }
        }
        // L1 constraints for s_y: t_y >= s_y and t_y >= -s_y (Ax >= l form)
        // -> -s_y + t_y >= 0
        // ->  s_y + t_y >= 0
        if (t_y_dim > 0)
        {
            for (int i = 0; i < s_y_dim; ++i)
            {
                // Row for: -s_y_i + t_y_i >= 0
                A_triplets.emplace_back(current_row, s_y_offset + i, -1.0);
                A_triplets.emplace_back(current_row, t_y_offset + i, 1.0);
                current_row++;
                // Row for: s_y_i + t_y_i >= 0
                A_triplets.emplace_back(current_row, s_y_offset + i, 1.0);
                A_triplets.emplace_back(current_row, t_y_offset + i, 1.0);
                current_row++;
            }
        }
        assert(current_row == m_n_constraints); // Sanity check final row count

        A_original.resize(m_n_constraints, m_n_variables);
        A_original.setFromTriplets(A_triplets.begin(), A_triplets.end());

        // --- Initialize unscaled Bounds Vectors (l, u) ---
        // These bounds remain in physical units even for the scaled problem
        m_lower_bound = Eigen::VectorXd(m_n_constraints);
        m_upper_bound = Eigen::VectorXd(m_n_constraints);
        m_b_eq_dynamic = Eigen::VectorXd::Zero(core_eq_rows);

        // Set static parts of bounds (inequalities) - dynamic parts (equalities, delta-u) set later
        current_row = core_eq_rows; // Reset row counter for bounds section
        // Input bounds
        if (m_useInputConstraints)
        {
            for (int i = 0; i < m_controlHorizon; ++i)
            {
                m_lower_bound.segment(current_row + i * m_u_dim, m_u_dim) = m_uMin;
                m_upper_bound.segment(current_row + i * m_u_dim, m_u_dim) = m_uMax;
            }
            current_row += input_ineq_rows;
        }
        // Output bounds
        if (m_useOutputConstraints)
        {
            for (int i = 0; i < m_predictionHorizon; ++i)
            {
                m_lower_bound.segment(current_row + i * m_y_dim, m_y_dim) = m_yMin;
                m_upper_bound.segment(current_row + i * m_y_dim, m_y_dim) = m_yMax;
            }
            current_row += output_ineq_rows;
        }
        // Delta-u bounds (dynamic part handled later)
        if (m_useDeltaInputConstraints)
        {
            m_lower_bound.segment(current_row, delta_input_ineq_rows).setConstant(-std::numeric_limits<double>::infinity()); // Placeholder
            m_upper_bound.segment(current_row, delta_input_ineq_rows).setConstant(std::numeric_limits<double>::infinity());  // Placeholder
            current_row += delta_input_ineq_rows;
        }
        // L1 bounds for g: Ax >= 0 => lower=0, upper=inf
        if (t_g_dim > 0)
        {
            m_lower_bound.segment(current_row, l1_g_ineq_rows).setConstant(0.0);
            m_upper_bound.segment(current_row, l1_g_ineq_rows).setConstant(std::numeric_limits<double>::infinity());
            current_row += l1_g_ineq_rows;
        }
        // L1 bounds for y: Ax >= 0 => lower=0, upper=inf
        if (t_y_dim > 0)
        {
            m_lower_bound.segment(current_row, l1_y_ineq_rows).setConstant(0.0);
            m_upper_bound.segment(current_row, l1_y_ineq_rows).setConstant(std::numeric_limits<double>::infinity());
            current_row += l1_y_ineq_rows;
        }
        assert(current_row == m_n_constraints); // Sanity check bounds filling

        // --- NORMALIZATION: Create Scaling Matrix and Scaled QP Matrices ---
        m_S = buildScalingMatrix();
        // m_S_inv = m_S.diagonal().cwiseInverse().asDiagonal();

        // H_scaled = S' * H_original * S (S'=S since diagonal)
        m_H_scaled = m_S * H_original * m_S;
        // f_scaled = S' * f_original
        m_f_scaled = m_S * f_original; // Base scaled gradient
        // A_scaled = A_original * S
        m_A_scaled = A_original * m_S;

        // Ensure sparse matrices passed to solver are compressed
        m_H_scaled.makeCompressed();
        m_A_scaled.makeCompressed();

        // --- Final Checks (Optional) ---
        if (static_cast<int>(m_A_scaled.rows()) != m_n_constraints || static_cast<int>(m_A_scaled.cols()) != m_n_variables)
        {
            throw std::runtime_error("Scaled constraint matrix A_scaled dimensions are incorrect.");
        }
        if (static_cast<int>(m_lower_bound.size()) != m_n_constraints || static_cast<int>(m_upper_bound.size()) != m_n_constraints)
        {
            throw std::runtime_error("Constraint bounds dimensions are incorrect.");
        }
        if (static_cast<int>(m_H_scaled.rows()) != m_n_variables || static_cast<int>(m_H_scaled.cols()) != m_n_variables)
        {
            throw std::runtime_error("Scaled Hessian matrix H_scaled dimensions are incorrect.");
        }
        if (static_cast<int>(m_f_scaled.size()) != m_n_variables)
        {
            throw std::runtime_error("Scaled gradient vector f_scaled dimensions are incorrect.");
        }

        m_is_problem_setup = true;
        m_is_solver_initialized = false; // Force re-initialization with new/scaled problem

    }

    void DDMPC::updateDynamicConstraints(Eigen::VectorXd u_ini, Eigen::VectorXd y_ini, const Eigen::VectorXd &reference, const Eigen::VectorXd &u_prev)
    {
        if (!m_is_problem_setup)
            throw std::runtime_error("setupProblem must be called first.");

        // --- Define dimensions and offsets ---
        const int g_dim = m_num_g;
        const int u_dim_pred = m_predictionHorizon * m_u_dim;
        const int y_dim_pred = m_predictionHorizon * m_y_dim;
        const int core_eq_rows = (m_Tini * (m_u_dim + m_y_dim)) + (u_dim_pred + y_dim_pred);

        // 1. Update equality constraint bounds (physical units)
        m_b_eq_dynamic.head(m_Tini * m_u_dim) = u_ini;
        m_b_eq_dynamic.segment(m_Tini * m_u_dim, m_Tini * m_y_dim) = y_ini;
        m_b_eq_dynamic.tail(core_eq_rows - (m_Tini * (m_u_dim + m_y_dim))).setZero(); // Uf*g - u = 0, Yf*g - y = 0

        m_lower_bound.head(core_eq_rows) = m_b_eq_dynamic;
        m_upper_bound.head(core_eq_rows) = m_b_eq_dynamic;

        // 2. Update inequality bounds (only delta-u depends on u_prev)
        int current_row = core_eq_rows;
        const int input_ineq_rows = m_useInputConstraints ? m_controlHorizon * m_u_dim : 0;
        current_row += input_ineq_rows; // Skip static bounds
        const int output_ineq_rows = m_useOutputConstraints ? m_predictionHorizon * m_y_dim : 0;
        current_row += output_ineq_rows; // Skip static bounds

        if (m_useDeltaInputConstraints)
        {
            const int delta_input_ineq_rows = m_controlHorizon * m_u_dim;
            if (u_prev.size() != m_u_dim)
                throw std::invalid_argument("u_prev size mismatch.");

            // NORMALIZATION: Note A_scaled = A_original * S. The constraint row for
            // u_i - u_{i-1} in A_original has +1 at u_i and -1 at u_{i-1}.
            // In A_scaled, this row becomes +scale_u at u_i_scaled and -scale_u at u_{i-1}_scaled.
            // The bounds l, u are physical limits on (u_i - u_{i-1}).
            // The solver uses l <= (A_scaled * x_scaled)_row <= u.
            // The bounds l, u SHOULD remain in physical units. TODO: verify this.

            int delta_u_row_offset = 0;
            for (int i = 0; i < m_controlHorizon; ++i)
            {
                for (int j = 0; j < m_u_dim; ++j)
                {
                    int row_idx = current_row + delta_u_row_offset;
                    if (i == 0)
                    {
                        // Constraint is u_0 - u_prev
                        // For i=0: Row has +1 at u_0, (u_prev is handled by bounds)
                        //          Row in A_scaled has +scale_u at u_0_scaled.
                        // We want l <= (A_scaled * x_scaled)_row <= u
                        // We want delta_min <= u_0 - u_prev <= delta_max.
                        // The row calculates u_0 = scale_u * u_0_scaled.
                        // We need bounds l', u' such that l' <= scale_u * u_0_scaled <= u'
                        // where l' = delta_min + u_prev[j] and u' = delta_max + u_prev[j].
                        m_lower_bound[row_idx] = m_deltaUmin[j] + u_prev[j];
                        m_upper_bound[row_idx] = m_deltaUmax[j] + u_prev[j];
                    }
                    else
                    {
                        // Constraint is u_i - u_{i-1}
                        // Row in A_orig has +1 at u_i, -1 at u_{i-1}.
                        // Row in A_scaled has +scale_u at u_i_scaled, -scale_u at u_{i-1}_scaled.
                        // This calculates scale_u * (u_i_scaled - u_{i-1}_scaled) = u_i - u_{i-1}.
                        // We want delta_min <= u_i - u_{i-1} <= delta_max.
                        m_lower_bound[row_idx] = m_deltaUmin[j];
                        m_upper_bound[row_idx] = m_deltaUmax[j];
                    }
                    delta_u_row_offset++;
                }
            }
            current_row += delta_input_ineq_rows;
        }
        // Skip L1 bounds (static)
    }

    void DDMPC::updateObjective(const Eigen::VectorXd &reference)
    {
        if (!m_is_problem_setup)
            throw std::runtime_error("setupProblem must be called first.");
        if (reference.size() != 0 && reference.size() != m_y_dim)
            throw std::invalid_argument("Reference size mismatch.");

        // --- Define dimensions and offsets ---
        const int y_dim_pred = m_predictionHorizon * m_y_dim;
        const int y_offset = m_num_g + m_predictionHorizon * m_u_dim; // Recalculate or get from members

        // --- Update Original Gradient (f_original) ---
        // Start with base gradient (L1 terms)
        Eigen::VectorXd f_original_updated = m_f_original_base;

        if (reference.size() > 0)
        {
            Eigen::VectorXd ref_expanded(y_dim_pred);
            for (int i = 0; i < m_predictionHorizon; ++i)
            {
                ref_expanded.segment(i * m_y_dim, m_y_dim) = reference;
            }

            // Build Q_expanded efficiently (block diagonal sparse)
            Eigen::SparseMatrix<double> Q_expanded_sparse(y_dim_pred, y_dim_pred);
            std::vector<Eigen::Triplet<double>> Q_exp_triplets;
            // Reserve assuming Q is somewhat sparse
            Q_exp_triplets.reserve(m_predictionHorizon * m_y_dim * m_y_dim / 2);
            for (int i = 0; i < m_predictionHorizon; ++i)
            {
                for (int r = 0; r < m_y_dim; ++r)
                {
                    for (int c = 0; c < m_y_dim; ++c)
                    {
                        if (m_Q_vec[i](r, c) != 0.0)
                        {
                            Q_exp_triplets.emplace_back(i * m_y_dim + r, i * m_y_dim + c, m_Q_vec[i](r, c));
                        }
                    }
                }
            }
            Q_expanded_sparse.setFromTriplets(Q_exp_triplets.begin(), Q_exp_triplets.end());

            // Update the part of f_original related to reference: -Q' * ref
            f_original_updated.segment(y_offset, y_dim_pred) = -Q_expanded_sparse.transpose() * ref_expanded;
        }

        // --- NORMALIZATION: Calculate Scaled Gradient (f_scaled) ---
        // f_scaled = S' * f_original_updated
        m_f_scaled = m_S.transpose() * f_original_updated;

        // --- NORMALIZATION: Update Scaled Hessian if Q/R changed ---
        // Check if Q or R actually changed since last setupProblem call
        // If yes, rebuild H_original and then H_scaled = S' * H_original * S
        bool weights_changed = false;
        if (weights_changed && m_is_problem_setup)
        {
            // Rebuild H_original (similar to setupProblem)
            Eigen::SparseMatrix<double> H_original_new(m_n_variables, m_n_variables);
            // TODO: fill H_original_new using m_Q_vec, m_R_vec, m_lambda_u ...

            // Then scale
            m_H_scaled = m_S.transpose() * H_original_new * m_S;
            m_H_scaled.makeCompressed();
            if (m_is_solver_initialized)
            {
                m_solver->updateHessianMatrix(m_H_scaled);
            }
        }

        // --- Update Solver ---
        // Gradient update is always needed if reference changed or first time
        if (m_is_solver_initialized)
        {
            m_solver->updateGradient(m_f_scaled);
            // Hessian update handled above if necessary
        }
    }

} // namespace DataDrivenMPC