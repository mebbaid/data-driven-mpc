#ifndef QP_SOLVER_H
#define QP_SOLVER_H

#include <Eigen/Dense>
#include <vector>

namespace DataDrivenMPC
{

    /**
 * @brief Abstract base class for a QP solver interface.
 *        Provides a common interface for different QP solvers.
 */
class QPSolver {
public:
    /**
     * @brief Virtual destructor.  Important for proper cleanup if using polymorphism.
     */
    virtual ~QPSolver() = default;

    /**
     * @brief Sets the quadratic objective function.
     * @param P The Hessian matrix (must be positive semi-definite).
     * @param q The linear term vector.
     */
    virtual void setObjective(const Eigen::MatrixXd& P, const Eigen::VectorXd& q) = 0;

    /**
     * @brief Sets the equality constraints.
     * @param A The equality constraint matrix.
     * @param b The equality constraint vector.
     */
    virtual void setEqualityConstraints(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) = 0;

    /**
     * @brief Sets the inequality constraints.
     * @param C The inequality constraint matrix.
     * @param l The lower bound vector.
     * @param u The upper bound vector.
     */
    virtual void setInequalityConstraints(const Eigen::MatrixXd& C, const Eigen::VectorXd& l, const Eigen::VectorXd& u) = 0;

    /**
     * @brief Solves the quadratic program.
     * @param x_optimal Output parameter: The optimal solution vector.
     * @return True if the solver was successful, false otherwise.
     */
    virtual bool solve(Eigen::VectorXd& x_optimal) = 0;

};

} // namespace DataDrivenMPC

#endif // QP_SOLVER_H