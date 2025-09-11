#include "data-driven-mpc/data-driven-mpc.h"
#include <iostream>

#include <Eigen/QR>

namespace DataDrivenMPC
{

    HankelMatrix::HankelMatrix(const std::vector<double> &data, int horizonLength)
        : m_horizonLength(horizonLength)
    {
        if (data.size() < static_cast<size_t>(horizonLength))
        {
            throw std::invalid_argument("Data length is insufficient for the given horizon length.");
        }

        int numCols = data.size() - horizonLength + 1;
        m_hankelMatrix.resize(horizonLength, numCols);

        for (int i = 0; i < horizonLength; ++i)
        {
            for (int j = 0; j < numCols; ++j)
            {
                m_hankelMatrix(i, j) = data[i + j];
            }
        }
    }

    HankelMatrix::HankelMatrix(const std::vector<Eigen::VectorXd> &data, int horizonLength)
        : m_horizonLength(horizonLength)
    {
        if (data.size() < static_cast<size_t>(horizonLength))
        {
            throw std::invalid_argument("Data length is insufficient for the given horizon length.");
        }

        int numCols = data.size() - horizonLength + 1;
        int numRowsInVector = data[0].rows(); // Assume all vectors have the same size
        m_hankelMatrix.resize(horizonLength * numRowsInVector, numCols);

        for (int i = 0; i < horizonLength; ++i)
        {
            for (int j = 0; j < numCols; ++j)
            {
                m_hankelMatrix.block(i * numRowsInVector, j, numRowsInVector, 1) = data[i + j];
            }
        }
    }

    const Eigen::MatrixXd &HankelMatrix::getMatrix() const
    {
        return m_hankelMatrix;
    }

    bool HankelMatrix::isPersistentlyExciting(int order, double tolerance) const
    {
        // --- Input Validation ---
        int max_possible_rank = std::min(m_hankelMatrix.rows(), m_hankelMatrix.cols());
        if (order < 0)
        {
            throw std::invalid_argument("Required rank ('order') cannot be negative.");
        }

        if (order > m_hankelMatrix.rows())
        {
            std::cerr << "Warning: Requested order (" << order
                      << ") for PE check exceeds Hankel matrix row dimension ("
                      << m_hankelMatrix.rows() << "). This check might not be meaningful." << std::endl;
            // Depending on interpretation, you might throw or just return false here
        }

        if (tolerance <= 0.0)
        {
            throw std::invalid_argument("Tolerance for PE check must be positive.");
        }

        // Handle empty matrix case
        if (m_hankelMatrix.rows() == 0 || m_hankelMatrix.cols() == 0)
        {
            // An empty matrix has rank 0. It's PE only if the required rank is 0.
            return (order == 0);
        }

        // --- Use ColPivHouseholderQR for Rank Computation ---
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(m_hankelMatrix);

        // Get the absolute values of the diagonal elements of the R factor.
        // The rank is the number of diagonal elements whose magnitude is > tolerance.
        // Eigen's internal representation might store R differently, but diagonal() on matrixQR() gives access.
        // Size of diagonal is min(rows, cols).
        const auto &R_diag_abs = qr.matrixQR().diagonal().cwiseAbs();

        // Count the number of diagonal elements above the *absolute* tolerance

        int computed_rank = 0;
        for (int i = 0; i < R_diag_abs.size(); ++i)
        {
            if (R_diag_abs(i) > tolerance)
            {
                computed_rank++;
            }
        }

        // Check if the computed rank meets the required rank 'order'
        return computed_rank >= order;
    }

} // namespace DataDrivenMPC
