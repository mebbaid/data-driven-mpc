import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_data(filename="simulation_results.csv"):
    """Reads simulation data from a CSV and generates plots."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{filename}': {e}")
        sys.exit(1)

    time_vec = df['Time']
    y_vec = df['Output']
    u_vec = df['Input']
    ref_vec = df['Reference']

    plt.style.use('ggplot') # Option 2

    # Plot Output and Reference
    plt.figure(figsize=(10, 6))
    plt.plot(time_vec, y_vec, label="Output (y)", marker='.', linestyle='-')
    plt.plot(time_vec, ref_vec, label="Reference", linestyle='--', color='red')
    plt.xlabel("Time Step (k)")
    plt.ylabel("Output Value")
    plt.title("System Output and Reference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_plot_py.png")
    print("Saved output plot to output_plot_py.png")
    plt.show() # Optionally show if needed interactively

    # Plot Control Input
    plt.figure(figsize=(10, 6))
    plt.plot(time_vec, u_vec, label="Control Input (u)", marker='.', linestyle='-', color='green')
    plt.xlabel("Time Step (k)")
    plt.ylabel("Input Value")
    plt.title("Control Input")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("input_plot_py.png")
    print("Saved input plot to input_plot_py.png")
    plt.show() # Optionally show if needed interactively

if __name__ == "__main__":
    # Allows specifying a different filename from command line if needed
    if len(sys.argv) > 1:
        plot_data(sys.argv[1])
    else:
        plot_data() # Default filename