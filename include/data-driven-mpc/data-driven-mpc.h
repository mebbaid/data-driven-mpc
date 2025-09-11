// data-driven-mpc.h
#ifndef DATA_DRIVEN_MPC_H
#define DATA_DRIVEN_MPC_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <memory>

#include "data-driven-mpc/qpsolver.h"

namespace DataDrivenMPC
{

    class HankelMatrix
    {
    public:
        /**
         * @brief Constructor: Builds the Hankel matrix from a data sequence.
         * @param data The input data sequence (std::vector).
         * @param horizonLength The horizon length (L) of the Hankel matrix.
         * @throws std::invalid_argument If the data is too short for the given horizon.
         */
        HankelMatrix(const std::vector<double> &data, int horizonLength);

        /**
         * @brief Constructor: Builds the Hankel matrix from a multi-dimensional data sequence.
         * @param data The input data sequence (std::vector of Eigen::VectorXd).
         * @param horizonLength The horizon length (L) of the Hankel matrix.
         * @throws std::invalid_argument If the data is too short for the given horizon.
         */
        HankelMatrix(const std::vector<Eigen::VectorXd> &data, int horizonLength);

        /**
         * @brief Gets the underlying Eigen matrix.
         * @return A constant reference to the Eigen::MatrixXd.
         */
        const Eigen::MatrixXd &getMatrix() const;

        /**
         * @brief Checks if the data sequence used to build the Hankel matrix is persistently exciting.
         * @param order The order of persistent excitation to check.
         * @param tolerance A tolerance for singular values (values below this are considered zero).
         * @return True if the data is likely PE of the given order, false otherwise.
         */
        bool isPersistentlyExciting(int order, double tolerance = 1e-6) const;

        /**
         * @brief Returns the horizon length (L) of the Hankel matrix.
         */
        int getHorizonLength() const { return m_horizonLength; }

        /**
         * @brief Returns the number of rows in the Hankel matrix.
         */
        int rows() const { return m_hankelMatrix.rows(); }

        /**
         * @brief Returns the number of columns in the Hankel matrix.
         */
        int cols() const { return m_hankelMatrix.cols(); }

    private:
        Eigen::MatrixXd m_hankelMatrix; // The actual Hankel matrix
        int m_horizonLength;            // The horizon length (L)
    };

    class DDMPC
    {
    public:
        /**
         * @brief Constructor for the Data-Driven Model Predictive Controller (DDMPC).
         * @param Tini The initial data length (Tini).
         * @param predictionHorizon The prediction horizon (N).
         * @param controlHorizon The control horizon (M).
         * @param Q The output weight matrix (Q).
         * @param R The input weight matrix (R).
         * @param solver The QP solver instance.
         * @param lambda_u Regularization weight for input changes.
         * @param lambda_y Regularization weight for output deviations.
         * @param lambda_g Regularization weight for the slack variables.
         * @throws std::invalid_argument If the horizons are invalid or the matrices have incorrect dimensions.
         */
        DDMPC(int Tini, int predictionHorizon, int controlHorizon,
              const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, std::unique_ptr<QPSolver> solver,
              std::string solver_name = "osqp",
              double lambda_u = 0.0, double lambda_y = 0.0, double lambda_g = 0.0,
              double scale_g = 1.0, double scale_u = 1.0, double scale_y = 1.0);

        /**
         * @brief Sets the input constraints for the DDMPC.
         * @param u_min The minimum input vector.
         * @param u_max The maximum input vector.
         * @throws std::invalid_argument If the input constraint vectors have incorrect dimensions.
         */
        void setInputConstraints(const Eigen::VectorXd &u_min, const Eigen::VectorXd &u_max);

        /**
         * @brief Sets the output constraints for the DDMPC.
         * @param y_min The minimum output vector.
         * @param y_max The maximum output vector.
         * @throws std::invalid_argument If the output constraint vectors have incorrect dimensions.
         */
        void setOutputConstraints(const Eigen::VectorXd &y_min, const Eigen::VectorXd &y_max);

        /**
         * @brief Sets the delta input constraints for the DDMPC.
         * @param delta_u_min The minimum change in input vector.
         * @param delta_u_max The maximum change in input vector.
         * @throws std::invalid_argument If the delta input constraint vectors have incorrect dimensions.
         */
        void setDeltaInputConstraints(const Eigen::VectorXd &delta_u_min, const Eigen::VectorXd &delta_u_max);

        /**
         * @brief Updates the weight matrices Q and R for the DDMPC.
         * @param Q The new output weight matrix.
         * @param R The new input weight matrix.
         * @throws std::invalid_argument If the matrices have incorrect dimensions.
         */
        void updateWeights(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R);

        /**
         * @brief Updates the weight matrices Q and R for the DDMPC.
         * @param Q_vec The new output weight matrices (std::vector of Eigen::MatrixXd).
         * @param R_vec The new input weight matrices (std::vector of Eigen::MatrixXd).
         * @throws std::invalid_argument If the matrices have incorrect dimensions.
         */
        void updateWeights(const std::vector<Eigen::MatrixXd> &Q_vec, const std::vector<Eigen::MatrixXd> &R_vec);

        /**
         * @brief Updates the regularization weights for the DDMPC.
         * @param lambda_u Regularization weight for input changes.
         * @param lambda_y Regularization weight for output deviations.
         * @param lambda_g Regularization weight for the slack variables.
         */
        void updateRegularizationWeights(double lambda_u, double lambda_y, double lambda_g);

        /**
         * @brief Sets the regularization weight for input changes.
         * @param lambda_u Regularization weight for input changes.
         * @throws std::invalid_argument If the weight is negative.
         */
        void setLambdaU(double lambda_u);

        /**
         * @brief Sets the regularization weight for output deviations.
         * @param lambda_y Regularization weight for output deviations.
         * @throws std::invalid_argument If the weight is negative.
         */
        void setLambdaY(double lambda_y);

        /**
         * @brief Sets the regularization weight for the slack variables.
         * @param lambda_g Regularization weight for the slack variables.
         * @throws std::invalid_argument If the weight is negative.
         */
        void setLambdaG(double lambda_g);

        /**
         * @brief Gets the regularization weight for input changes.
         * @return The regularization weight for input changes.
         */
        double getLambdaU() const;

        /**
         * @brief Gets the regularization weight for output deviations.
         * @return The regularization weight for output deviations.
         */
        double getLambdaY() const;

        /**
         * @brief Gets the regularization weight for the slack variables.
         * @return The regularization weight for the slack variables.
         */
        double getLambdaG() const;

        /**
         * @brief Builds the diagonal scaling matrix S for normalization.
         * @return The diagonal scaling matrix S.
         */
        Eigen::SparseMatrix<double> buildScalingMatrix() const;


        /**
         * @brief Solves the Data-Driven Model Predictive Control problem.
         * @param u_data The input data sequence (std::vector of Eigen::VectorXd).
         * @param y_data The output data sequence (std::vector of Eigen::VectorXd).
         * @param reference The reference trajectory (Eigen::VectorXd).
         * @param u_prev The previous control input (Eigen::VectorXd).
         * @param u_data_ini The initial input data sequence (std::vector of Eigen::VectorXd).
         * @param y_data_ini The initial output data sequence (std::vector of Eigen::VectorXd).
         * @return The optimal control input sequence (Eigen::VectorXd).
         * @throws std::invalid_argument If the input data sequences are too short.
         */
        Eigen::VectorXd solve(const std::vector<Eigen::VectorXd> &u_data,
                              const std::vector<Eigen::VectorXd> &y_data,
                              const Eigen::VectorXd &reference,
                              const Eigen::VectorXd &u_prev,
                              const std::vector<Eigen::VectorXd> &u_data_ini,
                              const std::vector<Eigen::VectorXd> &y_data_ini);

    private:
        std::unique_ptr<QPSolver> m_solver;
        std::string m_solver_name;

        // Parameters
        int m_Tini;
        int m_predictionHorizon;
        int m_controlHorizon;
        Eigen::MatrixXd m_Q;
        Eigen::MatrixXd m_R;
        std::vector<Eigen::MatrixXd> m_Q_vec;
        std::vector<Eigen::MatrixXd> m_R_vec;

        // Constraint flags and vectors
        bool m_useInputConstraints = false;
        bool m_useOutputConstraints = false;
        bool m_useDeltaInputConstraints = false;
        Eigen::VectorXd m_uMin;
        Eigen::VectorXd m_uMax;
        Eigen::VectorXd m_yMin;
        Eigen::VectorXd m_yMax;
        Eigen::VectorXd m_deltaUmin;
        Eigen::VectorXd m_deltaUmax;

        // Regularization weights
        double m_lambda_u;
        double m_lambda_y;
        double m_lambda_g;

        // QP problem dimensions
        int m_num_g;
        int m_u_dim;
        int m_y_dim;
        int m_n_variables;
        int m_n_constraints;

        // Auxiliary variables for L_1 norm terms
        int m_num_t_g;
        int m_num_t_y;

        // --- Optimized QP Matrices (Sparse) ---
        Eigen::SparseMatrix<double> m_H; // Hessian
        Eigen::VectorXd m_f;             // Gradient
        Eigen::SparseMatrix<double> m_A; // Constraint matrix
        Eigen::VectorXd m_lower_bound;   // Lower bound
        Eigen::VectorXd m_upper_bound;   // Upper bound

        // --- Pre-allocated Matrices for Efficiency ---
        Eigen::MatrixXd m_Q_expanded;     // Expanded Q (pre-allocated)
        Eigen::MatrixXd m_R_expanded;     // Expanded R (pre-allocated)
        Eigen::MatrixXd m_A_eq_static;    // Static part of equality constraints
        Eigen::VectorXd m_b_eq_dynamic;   // Dynamic part of equality constraint RHS
        Eigen::MatrixXd m_A_ineq_static;  // Static part of the inequality.
        Eigen::VectorXd m_l_ineq_dynamic; // Dynamic part of the inequality (lower)
        Eigen::VectorXd m_u_ineq_dynamic; // Dynamic part of the inequality (upper)

        // Solver initialization flag
        bool m_is_solver_initialized = false;
        bool m_is_problem_setup = false; // Flag: Matrices pre-computed?

        //
        std::vector<Eigen::VectorXd> m_initial_u_data; // Store as vectors
        std::vector<Eigen::VectorXd> m_initial_y_data;
        // Internal helper methods
        void checkInputData(const std::vector<Eigen::VectorXd> &u_data, const std::vector<Eigen::VectorXd> &y_data) const;
        void checkConstraintDimensions() const;
        void setupProblem(const std::vector<Eigen::VectorXd> &u_data, const std::vector<Eigen::VectorXd> &y_data);
        void updateDynamicConstraints(Eigen::VectorXd u_ini, Eigen::VectorXd y_ini, const Eigen::VectorXd &reference, const Eigen::VectorXd &u_prev);

        void updateObjective(const Eigen::VectorXd &reference);

        // flag for Hankel matrix reconstruction
        bool m_data_requires_pe_check;

        double m_scale_g, m_scale_u, m_scale_y;
        Eigen::SparseMatrix<double> m_H_scaled;
        Eigen::VectorXd m_f_scaled;
        Eigen::SparseMatrix<double> m_A_scaled;
        Eigen::SparseMatrix<double> m_S;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> m_S_inv;
        Eigen::VectorXd m_f_original_base;
    };

} // namespace DataDrivenMPC

#endif // DATA_DRIVEN_MPC_H