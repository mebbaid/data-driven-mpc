//data-driven-mpc.h
#ifndef DATA_DRIVEN_MPC_H
#define DATA_DRIVEN_MPC_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <QpSolversEigen/QpSolversEigen.hpp>

namespace DataDrivenMPC {

    typedef QpSolversEigen::Solver QPSolver;

    // (HankelMatrix class remains the same)
    class HankelMatrix {
    public:
        /**
         * @brief Constructor: Builds the Hankel matrix from a data sequence.
         * @param data The input data sequence (std::vector).
         * @param horizonLength The horizon length (L) of the Hankel matrix.
         * @throws std::invalid_argument If the data is too short for the given horizon.
         */
        HankelMatrix(const std::vector<double>& data, int horizonLength);

        /**
         * @brief Constructor: Builds the Hankel matrix from a multi-dimensional data sequence.
         * @param data The input data sequence (std::vector of Eigen::VectorXd).
         * @param horizonLength The horizon length (L) of the Hankel matrix.
         * @throws std::invalid_argument If the data is too short for the given horizon.
         */
        HankelMatrix(const std::vector<Eigen::VectorXd>& data, int horizonLength);

        /**
         * @brief Gets the underlying Eigen matrix.
         * @return A constant reference to the Eigen::MatrixXd.
         */
        const Eigen::MatrixXd& getMatrix() const;

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
        Eigen::MatrixXd m_hankelMatrix;  // The actual Hankel matrix
        int m_horizonLength;             // The horizon length (L)
    };


    class DDMPC {
    public:
        /**
         * @brief Constructor for the DDMPC controller.
         * @param Tini The initialization horizon.
         * @param predictionHorizon The prediction horizon (N).
         * @param controlHorizon The control horizon (M).  Must be <= N.
         * @param Q The state cost weighting matrix (diagonal).
         * @param R The input cost weighting matrix (diagonal).
         * @param solver A pointer to a QP solver instance.
         * @throws std::invalid_argument If the horizons are invalid or the matrices have incorrect dimensions.
         */
        DDMPC(int Tini, int predictionHorizon, int controlHorizon,
            const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, QPSolver* solver);

        // (Other methods remain the same, except for solve and buildOptimizationProblem)
        void setInputConstraints(const Eigen::VectorXd& u_min, const Eigen::VectorXd& u_max);
        void setDeltaInputConstraints(const Eigen::VectorXd& delta_u_min, const Eigen::VectorXd& delta_u_max);
        void setOutputConstraints(const Eigen::VectorXd& y_min, const Eigen::VectorXd& y_max);
        void updateWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);
        Eigen::VectorXd solve(const std::vector<Eigen::VectorXd>& u_data,
                          const std::vector<Eigen::VectorXd>& y_data,
                          const Eigen::VectorXd& reference,
                          const Eigen::VectorXd& u_prev);
    private:

        // Store parameters
        //int m_horizonLength; // Removed.  Now using Tini + predictionHorizon
        int m_Tini;
        int m_predictionHorizon;
        int m_controlHorizon;
        Eigen::MatrixXd m_Q;
        Eigen::MatrixXd m_R;
        QPSolver* m_solver;


        // Constraints (optional)
        bool m_useInputConstraints = false;
        bool m_useOutputConstraints = false;
        bool m_useDeltaInputConstraints = false;
        Eigen::VectorXd m_uMin;
        Eigen::VectorXd m_uMax;
        Eigen::VectorXd m_yMin;
        Eigen::VectorXd m_yMax;
        Eigen::VectorXd m_deltaUmin;
        Eigen::VectorXd m_deltaUmax;


        // Internal helper methods (implementation in the .cpp file)
        // Remove buildOptimizationProblem.  It's no longer needed.
        //void buildOptimizationProblem(const Eigen::MatrixXd& Hu, const Eigen::MatrixXd& Hy,
        //                         const Eigen::VectorXd& up, const Eigen::VectorXd& yp,
        //                         const Eigen::VectorXd& reference, const Eigen::VectorXd& u_prev,
        //                         Eigen::SparseMatrix<double>& H,
        //                         Eigen::MatrixXd& f); //pass by reference
        void checkInputData(const std::vector<Eigen::VectorXd>& u_data, const std::vector<Eigen::VectorXd>& y_data) const;
    };

} // namespace DataDrivenMPC

#endif // DATA_DRIVEN_MPC_H