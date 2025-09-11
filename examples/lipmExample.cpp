#include "data-driven-mpc/data-driven-mpc.h" // Assuming this includes Eigen and your QP solver wrapper
#include <iostream>
#include <random>
#include <deque>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath> // For cos, sin

// time profiling (optional)
#include <chrono>

using namespace DataDrivenMPC;

// --- LIPM System Parameters (Keep these consistent) ---
const double G = 9.81;     // Gravity
const double Z_C = 0.8;    // Constant CoM height (meters)
const double DT = 0.01;    // Simulation and control time step (seconds)
const double OMEGA_LIP = std::sqrt(G / Z_C); // Natural frequency of LIPM

// --- Helper Function for Discrete LIPM Dynamics (Euler Integration) ---
// State: [x, y, vx, vy] (CoM position and velocity)
// Input: [px, py] (Center of Pressure relative to origin)
Eigen::VectorXd lipm_dynamics(const Eigen::VectorXd& current_state, const Eigen::VectorXd& cop_input) {
    if (current_state.size() != 4 || cop_input.size() != 2) {
        throw std::runtime_error("Invalid dimensions for LIPM dynamics.");
    }
    Eigen::VectorXd next_state(4);
    double x = current_state(0);
    double y = current_state(1);
    double vx = current_state(2);
    double vy = current_state(3);
    double px = cop_input(0);
    double py = cop_input(1);

    double acc_x = OMEGA_LIP * OMEGA_LIP * (x - px);
    double acc_y = OMEGA_LIP * OMEGA_LIP * (y - py);

    next_state(0) = x + vx * DT;
    next_state(1) = y + vy * DT;
    next_state(2) = vx + acc_x * DT;
    next_state(3) = vy + acc_y * DT;

    return next_state;
}

// --- Helper Function for Saving Multi-Dimensional Data (Keep as before) ---
void saveDataLIPM(const std::string& filename,
    const std::vector<double>& time_vec,
    const std::vector<Eigen::VectorXd>& y_vec, // state [x, y, vx, vy]
    const std::vector<Eigen::VectorXd>& u_vec, // input [px, py]
    const std::vector<Eigen::VectorXd>& ref_vec) // reference [x_ref, y_ref, vx_ref, vy_ref]
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing!" << std::endl;
        return;
    }
    outfile << std::fixed << std::setprecision(6);
    outfile << "Time,x,y,vx,vy,px,py,x_ref,y_ref,vx_ref,vy_ref\n";
    for (size_t i = 0; i < time_vec.size(); ++i) {
        outfile << time_vec[i];
        // State (y)
        outfile << "," << y_vec[i](0) << "," << y_vec[i](1) << "," << y_vec[i](2) << "," << y_vec[i](3);
        // Input (u)
        outfile << "," << u_vec[i](0) << "," << u_vec[i](1);
        // Reference (ref) - note: this will be the CIRCULAR reference during DeePC sim
        outfile << "," << ref_vec[i](0) << "," << ref_vec[i](1) << "," << ref_vec[i](2) << "," << ref_vec[i](3);
        outfile << "\n";
    }
    outfile.close();
    std::cout << "Saved LIPM simulation data to " << filename << std::endl;
}

// --- Helper Function for Circular Reference (Keep as before) ---
Eigen::VectorXd getCircularReference(double time, double radius, double omega_ref) {
    Eigen::VectorXd ref(4); // State dimension [x, y, vx, vy]
    ref(0) = radius * std::cos(omega_ref * time); // x_ref
    ref(1) = radius * std::sin(omega_ref * time); // y_ref
    ref(2) = -radius * omega_ref * std::sin(omega_ref * time); // vx_ref
    ref(3) = radius * omega_ref * std::cos(omega_ref * time);  // vy_ref
    return ref;
}

// --- Helper Function for Linear Reference (for Data Generation) ---
Eigen::VectorXd getLinearReference(double time, double duration,
                                   const Eigen::VectorXd& start_pos,
                                   const Eigen::VectorXd& end_pos) {
    Eigen::VectorXd ref(4); // State dimension [x, y, vx, vy]
    double progress = std::min(1.0, std::max(0.0, time / duration));

    // Linear interpolation for position
    ref.head(2) = start_pos + progress * (end_pos - start_pos);

    // Constant velocity required to cover the distance in the duration
    if (duration > 1e-6) {
        ref.segment(2, 2) = (end_pos - start_pos) / duration;
    } else {
        ref.segment(2, 2).setZero();
    }
    return ref;
}


int main() {
    try {
        // --- System and MPC Parameters (Keep as before) ---
        int u_dim = 2; // CoP: [px, py]
        int y_dim = 4; // State: [x, y, vx, vy]
        int Tini = 4;
        int predictionHorizon = 10; // START TUNING HERE
        int controlHorizon = 1;
        int T = 100; // Data length

        std::cout << "Using data length T = " << T << std::endl;

        // --- Noise Setup (Keep as before) ---
        std::random_device rd;
        std::mt19937 gen(rd());
        double noise_pos_stddev = 0.005;
        double noise_vel_stddev = 0.01;
        std::normal_distribution<double> noise_dist_pos(0.0, noise_pos_stddev);
        std::normal_distribution<double> noise_dist_vel(0.0, noise_vel_stddev);
        std::vector<std::vector<double>> noise_sequences(y_dim);
        int total_time_steps = T + 100; // Enough for data gen + sim
        for(int j=0; j<y_dim; ++j) {
            noise_sequences[j].resize(total_time_steps);
            auto& dist = (j < 2) ? noise_dist_pos : noise_dist_vel;
            for (int i = 0; i < total_time_steps; ++i) {
                noise_sequences[j][i] = dist(gen);
            }
        }

        // --- Pre-calculated LQR Gain Matrix K_lqr ---
        // PASTE THE OUTPUT FROM THE PYTHON SCRIPT HERE
        // Example (replace with your actual computed values):
        Eigen::MatrixXd K_lqr(2, 4);
        K_lqr << 11.6765787,  0.0,         2.16344086,  0.0,
                 0.0,         11.6765787,  0.0,         2.16344086;
        std::cout << "Using pre-calculated LQR Gain K_lqr:\n" << K_lqr << std::endl;


        // --- Generate Training Data (LQR Tracking Straight Line) ---
        std::cout << "Generating training data using LQR..." << std::endl;
        std::vector<Eigen::VectorXd> u_data_vec(T);
        std::vector<Eigen::VectorXd> y_data_vec(T);
        Eigen::VectorXd current_lqr_state(y_dim);
        current_lqr_state.setZero(); // Start at origin at rest

        // Define the straight line path for LQR
        Eigen::VectorXd start_pos(2); start_pos << 0.0, 0.0;
        Eigen::VectorXd end_pos(2);   end_pos   << 0.5, 0.3; // Move to (0.5, 0.3)
        double lqr_traj_duration = T * DT; // Cover the distance over the data generation period

        for (int i = 0; i < T; ++i) {
            double current_lqr_time = i * DT;

            // Get reference state for LQR at this time step
            Eigen::VectorXd x_ref_k = getLinearReference(current_lqr_time, lqr_traj_duration, start_pos, end_pos);

            // Calculate LQR control input u_k = u_ref - K * (x - x_ref)
            // u_ref is the CoP needed to hold the reference state (px=x_ref, py=y_ref)
            Eigen::VectorXd u_ref_k = x_ref_k.head(2); // CoP = target CoM pos for equilibrium
            Eigen::VectorXd error_state = current_lqr_state - x_ref_k;
            Eigen::VectorXd u_lqr = u_ref_k - K_lqr * error_state;

            // Store the calculated LQR input
            u_data_vec[i] = u_lqr;

            // Simulate true dynamics using LQR input
            Eigen::VectorXd next_state_true = lipm_dynamics(current_lqr_state, u_lqr);

            // Add noise to get "measured" state for data AND for next LQR feedback
            Eigen::VectorXd next_state_noisy = next_state_true;
            for(int j=0; j<y_dim; ++j) {
                next_state_noisy(j) += noise_sequences[j][i];
            }
            y_data_vec[i] = next_state_noisy;

            // Update state for next step (use noisy state for LQR feedback)
            current_lqr_state = next_state_noisy;
        }
        std::cout << "Training data generated." << std::endl;


        // --- DDMPC Setup (Keep tuning parameters from previous attempt or adjust) --- //
        Eigen::MatrixXd Q(y_dim, y_dim);
        Q.setIdentity();
        Q(0,0) = 1.0; Q(1,1) = 1.0; // Weight position tracking higher
        Q(2,2) = 0.1; Q(3,3) = 0.1;   // Weight velocity tracking lower

        Eigen::MatrixXd R(u_dim, u_dim);
        R.setIdentity();
        R *= 0.1; // Penalize CoP inputs

        auto solver = std::make_unique<QPSolver>();
        std::string solver_name = "proxqp"; // "osqp" or "proxqp"

        // Regularization weights (CRITICAL TUNING PARAMETERS)
        double lambda_u = 1.0;  // Penalty on initial input deviation s_u
        double lambda_y = 1.0;  // <<--- TRY STARTING LOW (e.g., 1.0, 10.0)
        double lambda_g = 1.0;  // Penalty on g coefficients (L1)

        // Scaling factors
        double scale_g = 1.0;
        double scale_u = 1.0;
        double scale_y = 1.0;
        std::cout << "Using Scaling Factors: g=" << scale_g << ", u=" << scale_u << ", y=" << scale_y << std::endl;

        DDMPC controller(Tini, predictionHorizon, controlHorizon, Q, R, std::move(solver), solver_name,
                         lambda_u, lambda_y, lambda_g, scale_g, scale_u, scale_y);

        // --- Set Constraints (Keep as before, adjust if needed) ---
        Eigen::VectorXd u_min(u_dim); u_min << -0.5, -0.5;
        Eigen::VectorXd u_max(u_dim); u_max << 0.5, 0.5;
        controller.setInputConstraints(u_min, u_max);
        Eigen::VectorXd y_min(y_dim); y_min << -2.0, -2.0, -2.0, -2.0;
        Eigen::VectorXd y_max(y_dim); y_max << 2.0, 2.0, 2.0, 2.0;
        controller.setOutputConstraints(y_min, y_max);


        // --- Simulation Loop (DeePC controlling the system to follow CIRCULAR path) ---
        int simulation_steps = 20;
        double radius_ref = 0.3;
        double omega_ref = M_PI / 2.0; // Completes circle in 4s

        std::vector<double> time_vec;
        std::vector<Eigen::VectorXd> sim_y_vec;
        std::vector<Eigen::VectorXd> sim_u_vec;
        std::vector<Eigen::VectorXd> sim_ref_vec;

        std::deque<Eigen::VectorXd> u_data_ini_deque;
        std::deque<Eigen::VectorXd> y_data_ini_deque;

        // Populate initial deque from the *end* of the LQR-generated data
        for (int i = T - Tini; i < T; ++i) {
            if (i >= 0 && i < T) {
                 u_data_ini_deque.push_back(u_data_vec[i]);
                 y_data_ini_deque.push_back(y_data_vec[i]);
            } else {
                 throw std::runtime_error("Index out of bounds during initial deque population.");
            }
        }

        Eigen::VectorXd u_prev = u_data_ini_deque.empty() ? Eigen::VectorXd::Zero(u_dim) : u_data_ini_deque.back();
        // Initialize simulation state from the last point of the LQR trajectory data
        Eigen::VectorXd current_sim_y = y_data_ini_deque.empty() ? Eigen::VectorXd::Zero(y_dim) : y_data_ini_deque.back();

        std::cout << "DeePC Simulation started (tracking circular path)..." << std::endl;

        for (int k = 0; k < simulation_steps; ++k) {
            double current_time = k * DT;

            std::vector<Eigen::VectorXd> u_data_ini_vec(u_data_ini_deque.begin(), u_data_ini_deque.end());
            std::vector<Eigen::VectorXd> y_data_ini_vec(y_data_ini_deque.begin(), y_data_ini_deque.end());

            // Get CIRCULAR reference for DeePC
            Eigen::VectorXd reference_k = getCircularReference(current_time, radius_ref, omega_ref);

            // Update weights if needed (keep constant for now)
            controller.updateWeights(Q, R);

            // --- Solve MPC ---
            auto solve_start = std::chrono::high_resolution_clock::now();
            Eigen::VectorXd u_opt = controller.solve(u_data_vec, y_data_vec, reference_k, u_prev, u_data_ini_vec, y_data_ini_vec);
            auto solve_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> solve_duration = solve_end - solve_start;
            // Print timing occasionally
            // if (k < 5 || k % 50 == 0) { std::cout << "Step " << k << " Solve time: " << solve_duration.count() << " ms" << std::endl; }

            // --- Simulate True System ---
            Eigen::VectorXd y_next_sim_true = lipm_dynamics(current_sim_y, u_opt);

            // Add *new* noise for next step's measurement
            Eigen::VectorXd y_next_sim_noisy = y_next_sim_true;
             for(int j=0; j<y_dim; ++j) {
                 y_next_sim_noisy(j) += noise_sequences[j][T + k]; // Use noise beyond training data
             }

            // --- Store Results ---
            time_vec.push_back(current_time);
            sim_y_vec.push_back(y_next_sim_noisy); // Store noisy state
            sim_u_vec.push_back(u_opt);
            sim_ref_vec.push_back(reference_k); // Store circular reference

            // --- Update State and Deques for Next Iteration ---
            u_data_ini_deque.pop_front();
            u_data_ini_deque.push_back(u_opt);
            y_data_ini_deque.pop_front();
            y_data_ini_deque.push_back(y_next_sim_noisy); // Update deque with noisy state

            current_sim_y = y_next_sim_true; // Propagate true state
            u_prev = u_opt;
        }
        std::cout << "Simulation finished." << std::endl;

        // --- Save Simulation Data ---
        saveDataLIPM("lipm_simulation_results.csv", time_vec, sim_y_vec, sim_u_vec, sim_ref_vec);


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }

    std::cout << "Program finished successfully." << std::endl;
    return 0;
}