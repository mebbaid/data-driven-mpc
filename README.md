
## Data-Driven Predictive Control (DDMPC) Library

This C++ library implements the Data-enabled Predictive Control (DeePC) algorithm, inspired by the paper "Data‐enabled predictive control: In the Shallows of the DeePC" by Coulson, Lygeros, and Dörfler.

It computes control actions for a system based purely on previously collected input/output data, without requiring an explicit system model, by solving a Quadratic Program (QP) at each time step.

### Dependencies

**Mandatory:**

*   **CMake** (>= 3.15): For building the project.
*   **C++17 Compiler**: A compiler supporting C++17 standard (e.g., GCC, Clang, MSVC).
*   **Eigen3**: Library for linear algebra.
    ```bash
    # Example (Ubuntu/Debian)
    sudo apt-get update && sudo apt-get install libeigen3-dev
    ```
*   **qp-solvers-eigen**: A C++ wrapper for various QP solvers.
    *   Follow installation instructions: [https://github.com/ami-iit/qpsolvers-eigen](https://github.com/ami-iit/qpsolvers-eigen)
    *   This library *itself* requires at least one backend QP solver to be installed (e.g., OSQP or PROXQP).
*   **A QP Solver**: At least one solver compatible with `qp-solvers-eigen`.
    *   **OSQP**: Recommended for performance in many MPC tasks. Follow [OSQP Installation Guide](https://osqp.org/docs/get_started).
    *   **PROXQP**: Another option. Follow [PROXQP Installation Guide](https://github.com/Simple-Robotics/proxsuite).

**Optional (for Python Bindings):**

*   **Python 3** (Interpreter & Development Headers): Required to build the Python module.
    ```bash
    # Example (Ubuntu/Debian)
    sudo apt-get update && sudo apt-get install python3 python3-dev
    ```
*   **NumPy**: Python package for numerical operations (used for Eigen interaction).
    ```bash
    pip install numpy
    ```
*   **pybind11**: Header-only library for creating Python bindings. It's often included as a submodule or fetched by CMake. If not, install it:
    ```bash
    pip install pybind11
    ```

### Build and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install -j
    ```


### Problem Solved

The library solves a specific formulation of the Data-enabled Predictive Control (DeePC) problem at each time step `k`, incorporating regularization terms often associated with robustness (e.g., to noise or unmodelled effects), closely related to the concepts in the [original DeePC paper](https://arxiv.org/pdf/1811.05890) but with specific L1/L2 choices.

Let:
*   `N` be the prediction horizon (`predictionHorizon`).
*   `M` be the control horizon (`controlHorizon`, `M <= N`).
*   `T_ini` be the length of the initial trajectory (`Tini`).
*   `u_ini = [u_{k-T_{ini}}^T, ..., u_{k-1}^T]^T` be the past `T_ini` inputs.
*   `y_ini = [y_{k-T_{ini}}^T, ..., y_{k-1}^T]^T` be the past `T_ini` outputs.
*   `u = [u_k^T, ..., u_{k+N-1}^T]^T` be the predicted future input sequence.
*   `y = [y_k^T, ..., y_{k+N-1}^T]^T` be the predicted future output sequence.
*   `r = [r_k^T, ..., r_{k+N-1}^T]^T` be the reference trajectory over the prediction horizon.
*   `U_p, Y_p, U_f, Y_f` be the block Hankel matrices constructed from historical data.
*   `Q_i, R_i` be the stage cost weighting matrices at step `k+i`.
*   `lambda_g, lambda_y, lambda_u` be non-negative regularization weights.

The optimization problem solved is:

**Minimize** over `g, u, y, s_u, s_y` (and internal L1 slack variables `t_g, t_y`):

$$
\begin{align*}
sum_{i=0}^{N-1} &[ \| y_{k+i} - r_{k+i} \|^2_{Q_{i}} + \| u_{k+i} \|^2_{R_{i}} ] \\
& + \lambda_u  \| s_u \|^2_2 + \lambda_g \| g \|_1 + \lambda_y  \| s_y \|_1
\end{align*}
$$


*Subject to:*

1.  *Data-Driven Dynamics & Initial Condition Matching:*
    *   `U_p * g = u_ini + s_u`   (Input initial condition matching, L2 penalty on `s_u`)
    *   `Y_p * g = y_ini + s_y`   (Output initial condition matching, L1 penalty on `s_y`)
    *   `U_f * g = u`             (Future inputs determined by `g`)
    *   `Y_f * g = y`             (Future outputs determined by `g`)

2.  *Input Constraints (Optional):*
    *   `u_min <= u_{k+i} <= u_max` for `i = 0...M-1`

3.  *Output Constraints (Optional):*
    *   `y_min <= y_{k+i} <= y_max` for `i = 0...N-1`

4.  *Delta Input Constraints (Optional):*
    *   `delta_u_min <= u_{k+i} - u_{k+i-1} <= delta_u_max` for `i = 0...M-1`
        *(where `u_{k-1}` is the input applied at the previous step, `u_prev`)*

*(Note: The L1 norms `||g||_1` and `||s_y||_1` are handled internally by introducing slack variables `t_g`, `t_y` and linear inequality constraints: `t_g >= g`, `t_g >= -g`, `t_y >= s_y`, `t_y >= -s_y`)*

The primary output (`u_optimal_now`) is the first segment `u_k` of the optimal future input sequence `u`.
### Usage Interface

The core interface involves initializing the `DDMPC` class and calling its `solve` method repeatedly.

#### C++ Interface
**Include and Initialize**

```cpp
#include "data-driven-mpc/data-driven-mpc.h" // Adjust path if not installed globally
#include <vector>
#include <string>
#include <Eigen/Core>

// --- User-defined Parameters ---
// int Tini, predictionHorizon, controlHorizon, u_dim, y_dim;
// Eigen::MatrixXd Q, R;
// std::string solver_name; // e.g., "osqp"
// double lambda_u, lambda_y, lambda_g;
// double scale_g, scale_u, scale_y;
// (Define these variables based on your system and tuning)

// --- Initialize Controller ---
// The QP solver is created and managed internally.
DataDrivenMPC::DDMPC controller(
    Tini, predictionHorizon, controlHorizon, Q, R, solver_name,
    lambda_u, lambda_y, lambda_g, scale_g, scale_u, scale_y
);

// --- Optionally Set Constraints ---
// controller.setInputConstraints(u_min, u_max);
// controller.setOutputConstraints(y_min, y_max);
// controller.setDeltaInputConstraints(delta_u_min, delta_u_max);
```

**Call `solve` in Control Loop**
```cpp
// --- User-provided Data at each time step k ---
// std::vector<Eigen::VectorXd> u_data_vec;      // Historical inputs (length T >= Tini+N)
// std::vector<Eigen::VectorXd> y_data_vec;      // Historical outputs (length T >= Tini+N)
// Eigen::VectorXd reference;                   // Reference for current step (size y_dim)
// Eigen::VectorXd u_prev;                      // Input u(k-1) (size u_dim)
// std::vector<Eigen::VectorXd> u_data_ini_vec;  // Inputs u(k-Tini)...u(k-1) (vector of size Tini)
// std::vector<Eigen::VectorXd> y_data_ini_vec;  // Outputs y(k-Tini)...y(k-1) (vector of size Tini)
// (Prepare these variables based on your system state and history)

Eigen::VectorXd u_optimal_now; // Stores the result (size u_dim)
try {
     u_optimal_now = controller.solve(
        u_data_vec,
        y_data_vec,
        reference,
        u_prev,
        u_data_ini_vec,
        y_data_ini_vec
     );
     // --- Use u_optimal_now ---

} catch (const std::exception& e) {
    // --- Handle solver failure ---
}
```

#### Python Interface
**Import and Initialize**
```python
import numpy as np
import py_ddmpc # Use the name defined in your python/CMakeLists.txt
# --- User-defined Parameters ---
# Tini, prediction_horizon, control_horizon, u_dim, y_dim = ...
# Q = np.identity(y_dim) * ...
# R = np.identity(u_dim) * ...
# solver_name = "osqp" # e.g., "osqp"
# lambda_u, lambda_y, lambda_g = ...
# scale_g, scale_u, scale_y = ...
# (Define these variables based on your system and tuning)

# --- Initialize Controller ---
controller = py_ddmpc.DDMPC(
    Tini=Tini,
    predictionHorizon=prediction_horizon,
    controlHorizon=control_horizon,
    Q=Q,
    R=R,
    solver_name=solver_name,
    lambda_u=lambda_u,
    lambda_y=lambda_y,
    lambda_g=lambda_g,
    scale_g=scale_g,
    scale_u=scale_u,
    scale_y=scale_y
)

# --- Optionally Set Constraints ---
# u_min = np.array([...])
# u_max = np.array([...])
# controller.set_input_constraints(u_min=u_min, u_max=u_max)
# controller.set_output_constraints(...)
# controller.set_delta_input_constraints(...)
```

**Call `solve` in Control Loop**
```python
# --- User-provided Data at each time step k ---
# u_data_vec = [...]      # List of 1D numpy arrays (historical inputs, length T >= Tini+N)
# y_data_vec = [...]      # List of 1D numpy arrays (historical outputs, length T >= Tini+N)
# reference = np.array([...]) # 1D numpy array (size y_dim)
# u_prev = np.array([...])    # 1D numpy array u(k-1) (size u_dim)
# u_data_ini_list = [...] # List of 1D numpy arrays (inputs u(k-Tini)...u(k-1), size Tini)
# y_data_ini_list = [...] # List of 1D numpy arrays (outputs y(k-Tini)...y(k-1), size Tini)
# (Prepare these variables based on your system state and history)

u_optimal_now = None # Stores the result (1D numpy array)
try:
    u_optimal_now = controller.solve(
        u_data=u_data_vec,
        y_data=y_data_vec,
        reference=reference,
        u_prev=u_prev,
        u_data_ini=u_data_ini_list,
        y_data_ini=y_data_ini_list
    )
    # --- Use u_optimal_now ---

except Exception as e:
    # --- Handle solver failure ---
    print(f"Error during solve: {e}")
```

### License
Copyright (c) 2025, Italian Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
