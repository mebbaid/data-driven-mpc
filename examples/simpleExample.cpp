// In main.cpp
#include "data-driven-mpc/data-driven-mpc.h"
#include <iostream>
#include <random>
#include <deque>
#include <vector>
// #include "matplotlibcpp.h"
#include <fstream>
#include <iomanip>

// time profiling
#include <chrono>

// namespace plt = matplotlibcpp;

using namespace DataDrivenMPC;

void saveData(const std::string& filename,
    const std::vector<double>& time_vec,
    const std::vector<double>& y_vec,
    const std::vector<double>& u_vec,
    const std::vector<double>& ref_vec)
{
std::ofstream outfile(filename);
if (!outfile.is_open()) {
std::cerr << "Error: Could not open file " << filename << " for writing!" << std::endl;
return;
}

// Set precision for floating point numbers
outfile << std::fixed << std::setprecision(6);

// Write header
outfile << "Time,Output,Input,Reference\n";

// Write data rows
for (size_t i = 0; i < time_vec.size(); ++i) {
outfile << time_vec[i] << ","
      << y_vec[i] << ","
      << u_vec[i] << ","
      << ref_vec[i] << "\n";
}
outfile.close();
std::cout << "Saved simulation data to " << filename << std::endl;
}

int main() {
    try {
        // --- System Parameters ---
        int Tini = 5;
        int predictionHorizon = 10;
        int controlHorizon = 1;
        int u_dim = 1;
        int y_dim = 1;

        // --- Calculate Minimum Data Length (T) ---
        int T = Tini + predictionHorizon + 15 * u_dim * (Tini + predictionHorizon);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(0.5);
        std::normal_distribution<double> noise_dist(0.0, 0.05);

        //Pre-generate noise
        std::vector<double> noise_sequence;
        for (int i = 0; i < T + 100; ++i) {
            noise_sequence.push_back(noise_dist(gen));
        }

        // --- Generate Data (PRBS) ---
        std::vector<Eigen::VectorXd> u_data_vec;
        std::vector<Eigen::VectorXd> y_data_vec;
        Eigen::VectorXd y_current(y_dim);
        y_current << 0.0;
        for (int i = 0; i < T; ++i) {
            Eigen::VectorXd u(u_dim);
            u << (dist(gen) ? 1.0 : -1.0);
            u_data_vec.push_back(u);
            Eigen::VectorXd y_next(y_dim);
            if (i == 0) { y_next = 0.8 * y_current + 0.5 * Eigen::VectorXd::Zero(u_dim); }
            else { y_next = 0.8 * y_current + 0.5 * u_data_vec[i - 1]; }
            y_next(0) += noise_sequence[i];
            y_data_vec.push_back(y_next);
            y_current = y_next;
        }

        // --- DDMPC Setup --- //
        Eigen::MatrixXd Q(y_dim, y_dim); Q.setIdentity(); Q *= 1.0;
        Eigen::MatrixXd R(u_dim, u_dim); R.setIdentity(); R *= 0.1;
        auto solver = std::make_unique<QPSolver>();
        std::string solver_name = "osqp"; // or "proxqp"
        double lambda_u = 1.0; double lambda_y = 100.0; double lambda_g = 1.0;
        double scale_g = 1.0; double scale_u = 1.0; double scale_y = 1.0;
        std::cout << "Using Scaling Factors: g=" << scale_g << ", u=" << scale_u << ", y=" << scale_y << std::endl;
        DDMPC controller(Tini, predictionHorizon, controlHorizon, Q, R, std::move(solver), solver_name,
                         lambda_u, lambda_y, lambda_g, scale_g, scale_u, scale_y);

        // --- Set Constraints ---
        Eigen::VectorXd u_min(u_dim); u_min << -1.0;
        Eigen::VectorXd u_max(u_dim); u_max << 1.0;
        controller.setInputConstraints(u_min, u_max);
        Eigen::VectorXd y_min(y_dim); y_min << -10.0;
        Eigen::VectorXd y_max(y_dim); y_max << 10.0;
        controller.setOutputConstraints(y_min, y_max);
        // Eigen::VectorXd delta_u_min(u_dim); delta_u_min << -2.0;
        // Eigen::VectorXd delta_u_max(u_dim); delta_u_max << 2.0;
        // controller.setDeltaInputConstraints(delta_u_min, delta_u_max);

        // --- Simulation Loop ---
        Eigen::VectorXd reference(y_dim); reference << 1.0;
        Eigen::VectorXd u_prev(u_dim); u_prev.setZero();
        std::deque<Eigen::VectorXd> u_data_ini_deque;
        std::deque<Eigen::VectorXd> y_data_ini_deque;
        int simulation_steps = 50;
        std::vector<double> time_vec; std::vector<double> y_vec;
        std::vector<double> u_vec; std::vector<double> ref_vec;

        for (int i = T - Tini; i < T; ++i) {
            if (i >= 0 && i < u_data_vec.size()){
                 u_data_ini_deque.push_back(u_data_vec[i]);
                 y_data_ini_deque.push_back(y_data_vec[i]);
            } else { throw std::runtime_error("Index out of bounds during init."); }
        }
        if (!u_data_ini_deque.empty()) { u_prev = u_data_ini_deque.back(); }
        Eigen::VectorXd current_sim_y = y_data_ini_deque.back();
        std::cout << "Simulation started..." << std::endl;

        for (int k = 0; k < simulation_steps; ++k) {
            std::vector<Eigen::VectorXd> u_data_ini_vec(u_data_ini_deque.begin(), u_data_ini_deque.end());
            std::vector<Eigen::VectorXd> y_data_ini_vec(y_data_ini_deque.begin(), y_data_ini_deque.end());

            // Time-Varying Q and R
            std::vector<Eigen::MatrixXd> Q_vec(predictionHorizon, Q);
            std::vector<Eigen::MatrixXd> R_vec(predictionHorizon, R);
            for (int i = 0; i < predictionHorizon; ++i) {
                if (i >= 3 && i <= 6) { Q_vec[i] *= 5.0; }
                else { Q_vec[i] *= 0.2; }
                 R_vec[i] *= 0.1;
            }
            controller.updateWeights(Q_vec, R_vec);

            // Solve MPC
            Eigen::VectorXd u_opt = controller.solve(u_data_vec, y_data_vec, reference, u_prev, u_data_ini_vec, y_data_ini_vec);

            // Simulate System
            Eigen::VectorXd y_next_sim(y_dim);
            y_next_sim = 0.8 * current_sim_y + 0.5 * u_opt;
            y_next_sim(0) += noise_sequence[T + k];

            // Store Results (in memory)
            time_vec.push_back(static_cast<double>(k)); // Store time as double
            y_vec.push_back(y_next_sim(0));
            u_vec.push_back(u_opt(0));
            ref_vec.push_back(reference(0));

            // Update state and deque
            u_data_ini_deque.pop_front(); u_data_ini_deque.push_back(u_opt);
            y_data_ini_deque.pop_front(); y_data_ini_deque.push_back(y_next_sim);
            current_sim_y = y_next_sim;
            u_prev = u_opt;

            // Change reference
            if (k == 25) { reference(0) = -1.0; }
        }
        std::cout << "Simulation finished." << std::endl;

        // --- Save Data ---
        saveData("simulation_results.csv", time_vec, y_vec, u_vec, ref_vec);


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Program finished successfully." << std::endl;
    return 0;
}