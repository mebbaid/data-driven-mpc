import pandas as pd
import matplotlib.pyplot as plt
import sys
import os # For checking file existence if needed

def plot_lipm_data(filename="lipm_simulation_results.csv"):
    """Reads LIPM simulation data from a CSV and generates informative plots."""
    try:
        # Check if file exists before attempting to read
        if not os.path.exists(filename):
             print(f"Error: File '{filename}' not found.")
             sys.exit(1)
        df = pd.read_csv(filename)
        # Verify expected columns exist (optional but good practice)
        expected_cols = ['Time', 'x', 'y', 'vx', 'vy', 'px', 'py', 'x_ref', 'y_ref', 'vx_ref', 'vy_ref']
        if not all(col in df.columns for col in expected_cols):
             print(f"Warning: CSV file '{filename}' might be missing expected columns.")
             print(f"Expected: {expected_cols}")
             print(f"Found: {list(df.columns)}")
             # Decide whether to exit or proceed cautiously
             # sys.exit(1)

    except FileNotFoundError: # Catch again just in case os.path check fails somehow
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{filename}': {e}")
        sys.exit(1)

    time_vec = df['Time']

    # --- Plot 1: XY CoM Trajectory ---
    plt.figure(figsize=(8, 8))
    plt.style.use('ggplot')
    plt.plot(df['x'], df['y'], label="Actual CoM Path", marker='.', linestyle='-', markersize=3)
    plt.plot(df['x_ref'], df['y_ref'], label="Reference CoM Path", linestyle='--', color='red')
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("CoM Trajectory Tracking (XY Plane)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is maintained for circular path
    plt.tight_layout()
    plt.savefig("lipm_com_trajectory_plot.png")
    print("Saved CoM trajectory plot to lipm_com_trajectory_plot.png")
    # plt.show()

    # --- Plot 2: CoM Position Components vs Time ---
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1) # Position X
    plt.plot(time_vec, df['x'], label="Actual X", linestyle='-')
    plt.plot(time_vec, df['x_ref'], label="Reference X", linestyle='--')
    plt.ylabel("X Position (m)")
    plt.title("CoM Position vs. Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2) # Position Y
    plt.plot(time_vec, df['y'], label="Actual Y", linestyle='-')
    plt.plot(time_vec, df['y_ref'], label="Reference Y", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lipm_com_position_time_plot.png")
    print("Saved CoM position vs time plot to lipm_com_position_time_plot.png")
    # plt.show()

    # --- Plot 3: CoM Velocity Components vs Time (Optional but useful) ---
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1) # Velocity X
    plt.plot(time_vec, df['vx'], label="Actual vX", linestyle='-')
    plt.plot(time_vec, df['vx_ref'], label="Reference vX", linestyle='--')
    plt.ylabel("X Velocity (m/s)")
    plt.title("CoM Velocity vs. Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2) # Velocity Y
    plt.plot(time_vec, df['vy'], label="Actual vY", linestyle='-')
    plt.plot(time_vec, df['vy_ref'], label="Reference vY", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Y Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lipm_com_velocity_time_plot.png")
    print("Saved CoM velocity vs time plot to lipm_com_velocity_time_plot.png")
    # plt.show()


    # --- Plot 4: Control Input (CoP) vs Time ---
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1) # CoP X
    plt.plot(time_vec, df['px'], label="CoP X (px)", linestyle='-', color='green')
    plt.ylabel("CoP X Position (m)")
    plt.title("Control Input (CoP) vs. Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2) # CoP Y
    plt.plot(time_vec, df['py'], label="CoP Y (py)", linestyle='-', color='purple')
    plt.xlabel("Time (s)")
    plt.ylabel("CoP Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lipm_cop_input_time_plot.png")
    print("Saved CoP input vs time plot to lipm_cop_input_time_plot.png")
    # plt.show()

    plt.close('all') # Close all figures if not showing them interactively

if __name__ == "__main__":
    # Allows specifying a different filename from command line if needed
    if len(sys.argv) > 1:
        plot_lipm_data(sys.argv[1])
    else:
        plot_lipm_data() # Default filename "lipm_simulation_results.csv"