#ifndef DATA_DRIVEN_MPC_H
#define DATA_DRIVEN_MPC_H

#include <Eigen/Dense>  // For linear algebra
#include <vector>
#include <stdexcept>


namespace DataDrivenMPC {

// Forward declaration of a QP solver interface (implementation details later)
class QPSolver;

/**
 * @brief Class to represent and manipulate Hankel matrices.
 */
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
     * @param horizonLength The horizon length (L) for the Hankel matrices.
     * @param predictionHorizon The prediction horizon (N).
     * @param controlHorizon The control horizon (M).  Must be <= N.
     * @param Q The state cost weighting matrix (diagonal).
     * @param R The input cost weighting matrix (diagonal).
     * @param solver A pointer to a QP solver instance.
     * @throws std::invalid_argument If the horizons are invalid or the matrices have incorrect dimensions.
     */
    DDMPC(int horizonLength, int predictionHorizon, int controlHorizon,
         const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, QPSolver* solver);

    /**
     * @brief Sets the input and output constraints.
     * @param u_min Lower bound on the control input (Eigen::VectorXd).
     * @param u_max Upper bound on the control input (Eigen::VectorXd).
     */
    void setInputConstraints(const Eigen::VectorXd& u_min, const Eigen::VectorXd& u_max);

    /**
     *  @brief Set the change in control input constraints
     *  @param delta_u_min Lower bound on the delta control input (Eigen::VectorXd)
     *  @param delta_u_max Upper bound on the delta control input (Eigen::VectorXd)
     */
    void setDeltaInputConstraints(const Eigen::VectorXd& delta_u_min, const Eigen::VectorXd& delta_u_max);


    /**
     * @brief Sets the output constraints.
     * @param y_min Lower bound on the output (Eigen::VectorXd).
     * @param y_max Upper bound on the output (Eigen::VectorXd).
     */
    void setOutputConstraints(const Eigen::VectorXd& y_min, const Eigen::VectorXd& y_max);

    /**
     * @brief Solves the DDMPC optimization problem at a single time step.
     * @param u_data Past input data (std::vector of Eigen::VectorXd).
     * @param y_data Past output data (std::vector of Eigen::VectorXd).
     * @param reference The desired reference trajectory (Eigen::VectorXd).  Length should be N.
     * @param u_prev The previous control input (Eigen::VectorXd).  For calculating delta_u.
     * @return The optimal control input at the current time step (Eigen::VectorXd).
     * @throws std::runtime_error If the QP solver fails or the data is insufficient.
     */
    Eigen::VectorXd solve(const std::vector<Eigen::VectorXd>& u_data,
                          const std::vector<Eigen::VectorXd>& y_data,
                          const Eigen::VectorXd& reference,
                          const Eigen::VectorXd& u_prev);

    /**
     * @brief Update Cost function weight matrices.
     * @param Q The state cost weighting matrix (diagonal).
     * @param R The input cost weighting matrix (diagonal).
     * @throws std::invalid_argument if matrices does not match required dimension.
     */
    void updateWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

private:

    // Store parameters
    int m_horizonLength;
    int m_predictionHorizon;
    int m_controlHorizon;
    Eigen::MatrixXd m_Q;
    Eigen::MatrixXd m_R;
    QPSolver* m_solver;   // Pointer to the QP solver

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
    void buildOptimizationProblem(const Eigen::MatrixXd& Hu, const Eigen::MatrixXd& Hy,
                                 const Eigen::VectorXd& up, const Eigen::VectorXd& yp,
                                 const Eigen::VectorXd& reference, const Eigen::VectorXd& u_prev,
                                 Eigen::VectorXd& g_optimal); // Example - you'll need more parameters for your QP solver
    void checkInputData(const std::vector<Eigen::VectorXd>& u_data, const std::vector<Eigen::VectorXd>& y_data) const;
};

} // namespace DataDrivenMPC

#endif // DATA_DRIVEN_MPC_H