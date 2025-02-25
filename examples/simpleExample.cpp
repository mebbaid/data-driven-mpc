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
        int Tini = 5;      // L  -> Tini
        int predictionHorizon = 3;  // N
        int controlHorizon = 3;    // M
        int u_dim = 1;
        int y_dim = 1;

        // --- Calculate Minimum Data Length (T) ---
        // The minimum data length formula you had is not for this DeePC formulation
        // You simply need enough data for the Hankel matrix of size (Tini+N)
        int T = Tini + predictionHorizon + 50; // Add extra data for persistent excitation

        // --- Generate Data ---
        std::vector<Eigen::VectorXd> u_data_vec;
        std::vector<Eigen::VectorXd> y_data_vec;

        // PRBS Generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(0.5);
        for (int i = 0; i < T; ++i) {
            Eigen::VectorXd u(u_dim);
            u << (dist(gen) ? 1.0 : -1.0);
            u_data_vec.push_back(u);

            Eigen::VectorXd y(y_dim);
            if (i == 0) {
                y << 0.0;
            } else {
                y = 0.8 * y_data_vec.back() + 0.5 * u_data_vec[i - 1];
            }
            y_data_vec.push_back(y);
        }

        // construct Hankel matrices to check dimensions
        HankelMatrix Hu(u_data_vec, Tini + predictionHorizon); // Use Tini + predictionHorizon
        HankelMatrix Hy(y_data_vec, Tini + predictionHorizon); // Use Tini + predictionHorizon
        std::cout << "Hu dimensions: " << Hu.rows() << " x " << Hu.cols() << std::endl;
        std::cout << "Hy dimensions: " << Hy.rows() << " x " << Hy.cols() << std::endl;

        // --- DDMPC Setup ---
        Eigen::MatrixXd Q(predictionHorizon, predictionHorizon); // Q is for the *prediction* horizon
        Q.setIdentity();
        Q *= 10; // You may want different weights for each output in MIMO
        Eigen::MatrixXd R(controlHorizon, controlHorizon);       // R is for the *control* horizon
        R.setIdentity();
        R *= 0.1;  // You may want different weights for each input in MIMO

        QpSolversEigen::Solver* solver = new QpSolversEigen::Solver;

        DDMPC controller(Tini, predictionHorizon, controlHorizon, Q, R, solver); // Tini, not horizonLength

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
        Eigen::VectorXd reference(predictionHorizon * y_dim);  // Expanded reference
        reference << 1.0, 1.0, 1.0;  // Example: Constant reference of 1.0 for all prediction steps.
        Eigen::VectorXd u_prev(u_dim); // Not used directly in the new implementation (for constraints)
        u_prev.setZero();

        // solve
        Eigen::VectorXd u_opt = controller.solve(u_data_vec, y_data_vec, reference, u_prev);
        std::cout << "Optimal Control Input: " << u_opt.transpose() << std::endl;

        delete solver;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}