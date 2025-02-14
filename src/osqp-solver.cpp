#include "osqp-solver.h"
#include <iostream>     // For debugging (optional)
#include <Eigen/Sparse> // Include the Eigen sparse matrix header!
#include "util.h"       // Include for osqp_update_lin_cost
#include "types.h"

namespace DataDrivenMPC
{

    OSQPSolver::OSQPSolver() : m_workspace(nullptr), m_data(nullptr), m_settings(nullptr)
    {
        // Initialize OSQP settings to default values.
        m_settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
        if (m_settings)
        {
            osqp_set_default_settings(m_settings);
            m_settings->verbose = false; // Disable verbose output
        }
    }

    OSQPSolver::~OSQPSolver()
    {
        // Free the OSQP workspace and data.
        if (m_workspace)
        {
            osqp_cleanup(m_workspace);
            m_workspace = nullptr;
        }
        if (m_settings)
        {
            c_free(m_settings);
            m_settings = nullptr;
        }
    }

    void OSQPSolver::setObjective(const Eigen::MatrixXd &P, const Eigen::VectorXd &q)
    {
        // Convert Eigen matrix P to a CSC matrix (OSQP's required format) and ensure it is compressed
        Eigen::SparseMatrix<double> P_sparse = P.sparseView();
        P_sparse.makeCompressed();
        // Update P and q in the OSQP workspace
        if (osqp_update_P(m_workspace, P_sparse.valuePtr(), OSQP_NULL, P_sparse.nonZeros()) < 0)
        {
            throw std::runtime_error("Error in setObjective: osqp_update_P failed.");
        }
        if (osqp_update_lin_cost(m_workspace, q.data()) < 0)
        {
            throw std::runtime_error("Error in setObjective: osqp_update_lin_cost failed.");
        }
    }

    void OSQPSolver::setEqualityConstraints(const Eigen::MatrixXd &A, const Eigen::VectorXd &b)
    {
        Eigen::SparseMatrix<double> A_sparse = A.sparseView();
        A_sparse.makeCompressed();
        // Update both A, l, and u
        if (osqp_update_A(m_workspace, A_sparse.valuePtr(), OSQP_NULL, A_sparse.nonZeros()) < 0)
        {
            throw std::runtime_error("Error in setEqualityConstraints: osqp_update_A failed.");
        }

        if (osqp_update_bounds(m_workspace, b.data(), b.data()) < 0)
        {
            throw std::runtime_error("Error in setEqualityConstraints: osqp_update_bounds failed.");
        }
    }

    void OSQPSolver::setInequalityConstraints(const Eigen::MatrixXd &C, const Eigen::VectorXd &l, const Eigen::VectorXd &u)
    {
        // Convert Eigen matrix C to a CSC matrix
        Eigen::SparseMatrix<double> C_sparse = C.sparseView();
        C_sparse.makeCompressed();

        // Update A (constraint matrix), l (lower bound), and u (upper bound) in the OSQP workspace
        if (osqp_update_A(m_workspace, C_sparse.valuePtr(), OSQP_NULL, C_sparse.nonZeros()) < 0)
        {
            throw std::runtime_error("Error in setInequalityConstraints: osqp_update_A failed.");
        }
        if (osqp_update_bounds(m_workspace, l.data(), u.data()) < 0)
        {
            throw std::runtime_error("Error in setInequalityConstraints: osqp_update_bounds failed.");
        }
    }

    bool OSQPSolver::solve(Eigen::VectorXd &x_optimal)
    {
        // OSQP should have been initialized in DDMPC::solve
        if (!m_workspace)
        {
            throw std::runtime_error("OSQP workspace not initialized.  Call DDPC::solve first.");
        }

        // Solve the QP
        if (osqp_solve(m_workspace) != 0)
        {
            std::cerr << "OSQP Solver Error: " << m_workspace->info->status << std::endl;
            return false;
        }

        // Check solver status
        if (m_workspace->info->status_val != OSQP_SOLVED)
        {
            std::cerr << "OSQP Solver Error: " << m_workspace->info->status << std::endl;
            return false;
        }

        // Copy the solution
        x_optimal.resize(m_data->n);
        for (int i = 0; i < m_data->n; ++i)
        {
            x_optimal(i) = m_workspace->solution->x[i];
        }
        return true;
    }

    void OSQPSolver::allocateProblemData(c_int n, c_int m)
    {
        // This method is not supposed to be called by the user. osqp setup should be called only once!
    }

    void OSQPSolver::freeProblemData()
    {
        // OSQP manages deallocation of problem data, so we don't need to manually free it here
    }

    void OSQPSolver::setWorkspace(OSQPWorkspace *workspace)
    {
        m_workspace = workspace;
    }

    void OSQPSolver::setData(OSQPData *data)
    {
        m_data = data;
    }

} // namespace DataDrivenMPC