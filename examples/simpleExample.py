import py_ddmpc
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    print("Testing DDMPC (C++ Example Style - with Time-Varying Q, R)...")
    try:
        # --- System Parameters ---
        Tini = 5
        N = 10  # Prediction horizon
        M = 1   # Control horizon.
        u_dim = 1
        y_dim = 1

        # --- Calculate Minimum Data Length (T) ---
        T = Tini + N + 15 * u_dim * (Tini + N)

        # --- Generate Data (PRBS) ---
        np.random.seed(42)
        u_data = [np.array([1.0 if np.random.rand() > 0.5 else -1.0], dtype=np.float64) for _ in range(T)]
        noise_sequence = np.random.normal(0, 0.05, T)

        y_data = []
        for i in range(T):
            if i == 0:
                y_k = np.array([0.0], dtype=np.float64)
            else:
                y_k = 0.8 * y_data[i-1] + 0.5 * u_data[i-1] + noise_sequence[i]
                y_k = np.array(y_k, dtype=np.float64)
            y_data.append(y_k)

        # --- DDMPC Setup ---
        Q = np.eye(y_dim, dtype=np.float64) * 1.0  # Initial Q
        R = np.eye(u_dim, dtype=np.float64) * 0.1  # Initial R
        lambda_u = 1.0  # Turn off input regularization
        lambda_y = 1.0  # Turn off output regularization
        lambda_g = 1.0

        scale_u = 1.0
        scale_y = 1.0
        scale_g = 1.0

        ddmpc = py_ddmpc.DDMPC(Tini, N, M, Q, R, lambda_u, lambda_y, lambda_g,
                               scale_u, scale_y, scale_g)

        # --- Set Constraints ---
        u_min = np.array([-5.0], dtype=np.float64)
        u_max = np.array([5.0], dtype=np.float64)
        ddmpc.set_input_constraints(u_min, u_max)

        y_min = np.array([-10.0], dtype=np.float64)
        y_max = np.array([10.0], dtype=np.float64)
        ddmpc.set_output_constraints(y_min, y_max)

        delta_u_min = np.array([-2.0], dtype=np.float64)
        delta_u_max = np.array([2.0], dtype=np.float64)
        ddmpc.set_delta_input_constraints(delta_u_min, delta_u_max)

        # --- Simulation Loop ---
        reference = np.array([1.0], dtype=np.float64)
        u_prev = np.array([0.0], dtype=np.float64)

        u_data_ini = [np.array(u, dtype=np.float64) for u in u_data[:Tini]]
        y_data_ini = [np.array(y, dtype=np.float64) for y in y_data[:Tini]]

        simulation_steps = 50
        time_vec = []
        y_vec = []
        u_vec = []
        ref_vec = []

        print("Simulation started...")
        for k in range(simulation_steps):
            start = time.time()

            # --- Time-Varying Q and R ---
            Q_vec = []  # Initialize as empty lists
            R_vec = []

            # Example:  Make Q larger in the middle, then smaller
            for i in range(N):
                if 3 <= i <= 6:
                    q_temp = Q * 5.0  # Create a *new* array
                else:
                    q_temp = Q * 0.2
                r_temp = R * 0.1
                Q_vec.append(q_temp)  # Append the *new* array
                R_vec.append(r_temp)

            Q_vec_eigen = [np.array(q, dtype=np.float64) for q in Q_vec]
            R_vec_eigen = [np.array(r, dtype=np.float64) for r in R_vec]

            # --- Update Weights ---
            ddmpc.update_weights(Q_vec_eigen, R_vec_eigen)

            # --- Solve --- #
            u_optimal = ddmpc.solve(
                [np.array(u, dtype=np.float64) for u in u_data],
                [np.array(y, dtype=np.float64) for y in y_data],
                np.array(reference, dtype=np.float64),
                np.array(u_prev, dtype=np.float64),
                [np.array(u, dtype=np.float64) for u in u_data_ini],
                [np.array(y, dtype=np.float64) for y in y_data_ini]
            )
            end = time.time()

            u_optimal = np.array(u_optimal, dtype=np.float64)
            y_k = 0.8 * y_data_ini[-1] + 0.5 * u_optimal + noise_sequence[Tini + k]
            y_k = np.array(y_k, dtype=np.float64)

            # Update
            u_prev = u_optimal
            u_data_ini.pop(0)
            u_data_ini.append(u_optimal)
            y_data_ini.pop(0)
            y_data_ini.append(y_k)

            time_vec.append(k)
            y_vec.append(y_k[0])
            u_vec.append(u_optimal[0])
            ref_vec.append(reference[0])

            if k == 25:
                reference = np.array([-1.0], dtype=np.float64)

        print("Simulation finished.")

        # Plotting
        plt.figure(figsize=(12, 7.8))
        plt.subplot(2, 1, 1)
        plt.plot(time_vec, y_vec, label="Output (y)")
        plt.plot(time_vec, ref_vec, label="Reference")
        plt.xlabel("Time Step (k)")
        plt.ylabel("Output")
        plt.title("System Output and Reference")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_vec, u_vec, label="Control Input (u)")
        plt.xlabel("Time Step (k)")
        plt.ylabel("Control Input")
        plt.title("Control Input")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except RuntimeError as e:
        print("Error in DDMPC:", e)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()