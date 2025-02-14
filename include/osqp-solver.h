#ifndef OSQP_SOLVER_H
#define OSQP_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <qp-solver.h>
#include <osqp.h>
#include  <types.h>

namespace DataDrivenMPC
{


/**
 * @brief Concrete implementation of the QPSolver interface using OSQP.
 */
class OSQPSolver : public QPSolver {
public:
    /**
     * @brief Constructor.
     */
    OSQPSolver();

    /**
     * @brief Destructor.  Frees the OSQP workspace.
     */
    ~OSQPSolver() override;

    // Implement the QPSolver interface methods.  These are now pure virtual.
    void setObjective(const Eigen::MatrixXd& P, const Eigen::VectorXd& q) override;
    void setEqualityConstraints(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) override;
    void setInequalityConstraints(const Eigen::MatrixXd& C, const Eigen::VectorXd& l, const Eigen::VectorXd& u) override;
    bool solve(Eigen::VectorXd& x_optimal) override;

    // set the osqp
    void setWorkspace(OSQPWorkspace* workspace);
    void setData(OSQPData* data);

    // Helper functions are no longer needed since osqp_setup is called in DDMPC::solve
    void updateProblemData();
    void freeProblemData();
    void allocateProblemData(c_int n, c_int m);


private:
    // OSQP data structures
    OSQPWorkspace* m_workspace = nullptr;  // Pointer to the OSQP workspace
    OSQPData* m_data = nullptr;         // Pointer to the OSQP data
    OSQPSettings* m_settings = nullptr;    // Pointer to the OSQP settings
    // No longer needed, since setup is done in DDMPC::solve
    // bool m_matricesInitialized = false;



};

} // namespace DataDrivenMPC

#endif // OSQP_SOLVER_H