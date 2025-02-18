#include "data-driven-mpc.h"
#include <iostream>

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
        if (order > m_hankelMatrix.rows())
        {
            throw std::invalid_argument("Order for persistency of excitation check cannot exceed Hankel matrix row dimension.");
        }
        // Use SVD to check for persistent excitation
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(m_hankelMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd &singularValues = svd.singularValues();

        // Count the number of singular values above the tolerance
        int count = 0;
        for (int i = 0; i < singularValues.size(); ++i)
        {
            if (singularValues(i) > tolerance)
            {
                count++;
            }
        }

        return count >= order;
    }

} // namespace DataDrivenMPC