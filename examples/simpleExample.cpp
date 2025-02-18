//main.cpp
// In main.cpp
#include "data-driven-mpc.h"
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <iostream>
#include <random>
#include <deque>

using namespace DataDrivenMPC;

int main() {
    try {
        // --- System Parameters ---
        int horizonLength = 5;      // L
        int predictionHorizon = 3;  // N
        int controlHorizon = 2;    // M
        int u_dim = 1;
        int y_dim = 1;

        // --- Calculate Minimum Data Length (T) ---
        int min_data_length = horizonLength + (horizonLength + predictionHorizon - 1) * (u_dim + y_dim) - 1;
        int T = min_data_length + 50;  // Add extra data

        // --- Generate Data ---
        std::vector<Eigen::VectorXd> u_data_vec;  // Use vectors for initial data generation
        std::vector<Eigen::VectorXd> y_data_vec;

        // PRBS Generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(0.5);
        for (int i = 0; i < T; ++i) {
            Eigen::VectorXd u(u_dim);
            u << (dist(gen) ? 1.0 : -1.0);
            u_data_vec.push_back(u);  // Push onto the vector

            Eigen::VectorXd y(y_dim);
            if (i == 0) {
                y << 0.0;
            } else {
                y = 0.8 * y_data_vec.back() + 0.5 * u_data_vec[i - 1];
            }
            y_data_vec.push_back(y); // Push onto the vector
        }

        // construct Hankel matrices to check
        HankelMatrix Hu(u_data_vec, horizonLength);
        HankelMatrix Hy(y_data_vec, horizonLength);
        std::cout << "Hu dimensions: " << Hu.rows() << " x " << Hu.cols() << std::endl;
        std::cout << "Hy dimensions: " << Hy.rows() << " x " << Hy.cols() << std::endl;
        // --- DDMPC Setup ---
        Eigen::MatrixXd Q(predictionHorizon, predictionHorizon);
        Q.setIdentity();
        Q *= 10;
        Eigen::MatrixXd R(controlHorizon, controlHorizon);
        R.setIdentity();
        R *= 0.1;

        QpSolversEigen::Solver* solver = new QpSolversEigen::Solver;

        DDMPC controller(horizonLength, predictionHorizon, controlHorizon, Q, R, solver);

        // --- Set Constraints (Optional) ---
        Eigen::VectorXd u_min(u_dim);
        u_min << -2.0;
        Eigen::VectorXd u_max(u_dim);
        u_max << 2.0;
        controller.setInputConstraints(u_min, u_max);

        Eigen::VectorXd y_min(y_dim);
        y_min << -5.0;
        Eigen::VectorXd y_max(y_dim);
        y_max << 5.0;

        controller.setOutputConstraints(y_min, y_max);

        Eigen::VectorXd delta_u_min(u_dim);
        delta_u_min << -1.0;
        Eigen::VectorXd delta_u_max(u_dim);
        delta_u_max << 1.0;
        controller.setDeltaInputConstraints(delta_u_min, delta_u_max);

        // --- Solve DDMPC ---
        Eigen::VectorXd reference(predictionHorizon * y_dim);
        reference.setZero();
        Eigen::VectorXd u_prev(u_dim);
        u_prev.setZero();

        Eigen::VectorXd u = controller.solve(u_data_vec, y_data_vec, reference, u_prev);
        std::cout << "Control input: " << u.transpose() << std::endl;

        delete solver;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}