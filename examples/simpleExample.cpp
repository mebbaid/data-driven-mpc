#include <iostream>
#include <ctime>
#include <vector>
#include <cstdlib>

#include <Eigen/Dense>
#include "data-driven-mpc.h"
#include "osqp-solver.h"


int main() {
    // --- System Parameters (Double Integrator) ---
    double dt = 0.1;  // Sampling time
    Eigen::MatrixXd A(2, 2);
    A << 1, dt,
         0, 1;
    Eigen::VectorXd B(2);
    B << 0.5 * dt * dt,
         dt;
    Eigen::MatrixXd C(1, 2);
    C << 1, 0;

    // --- Data Generation ---
    int T = 100;  // Length of data collection
    std::vector<Eigen::VectorXd> u_data;
    std::vector<Eigen::VectorXd> y_data;
    Eigen::VectorXd x(2);
    x.setZero();  // Initial state

    // Generate persistently exciting input (PRBS)
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // Seed the random number generator
    for (int k = 0; k < T; ++k) {
        // Generate a random input (-1 or 1) for a PRBS signal
        double u_val = (std::rand() % 2 == 0) ? -1.0 : 1.0;
        Eigen::VectorXd u(1);
        u << u_val;
        u_data.push_back(u);

        // Simulate the system
        x = A * x + B * u_val;
        Eigen::VectorXd y = C * x;
        y_data.push_back(y);
    }

     // --- DDMPC Setup ---
    int L = 10;  // Hankel matrix horizon length
    int N = 15;  // Prediction horizon
    int M = 5;   // Control horizon
    int inputDim = u_data[0].size();
    int outputDim = y_data[0].size();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(outputDim, outputDim);  // Output weight
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(inputDim, inputDim); // Input weight

    // // check persistance of excitation
    // DataDrivenMPC::HankelMatrix check(u_data, L);
    // if (!check.isPersistentlyExciting(L, 1e-6)) {
    //     std::cerr << "Data is not persistently exciting of order 2." << std::endl;
    //     return 1;
    // }


    // Create OSQP solver instance
    DataDrivenMPC::QPSolver* solver = new DataDrivenMPC::OSQPSolver();

    // Create DDPC controller instance
    DataDrivenMPC::DDMPC controller(L, N, M, Q, R, solver);

    // --- Control Loop ---
    int control_iterations = 50;
    Eigen::VectorXd u_prev(1);
    u_prev.setZero();  // Initialize previous input

    Eigen::VectorXd reference(outputDim * N); // TODO: is this the needed orrect size: outputDim * N
    for (int i = 0; i < N; ++i) {
        reference.segment(i * outputDim, outputDim) << 1.0; // Set each output to 1.0
    }

    std::cout << "Starting control loop...\n";
    for (int k = 0; k < control_iterations; ++k) {

        // Get the optimal control input from DDPC
        Eigen::VectorXd u_optimal = controller.solve(u_data, y_data, reference, u_prev);

        // Apply the control input to the *real* system (simulated here)
        x = A * x + B * u_optimal(0);
        Eigen::VectorXd y = C * x;  // Measure the output

        // Store data for next iteration (rolling window)
        u_data.push_back(u_optimal);
        y_data.push_back(y);
        u_prev = u_optimal;  // Update previous input

        // Print results (optional)
        std::cout << "k=" << k << ", u=" << u_optimal(0) << ", y=" << y(0) << std::endl;
        // Remove the oldest data point to maintain a constant window size
        u_data.erase(u_data.begin());
        y_data.erase(y_data.begin());
    }

    // Clean up
    delete solver;

    return 0;
}