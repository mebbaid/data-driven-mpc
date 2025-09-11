import matplotlib.pyplot as plt
import numpy as np

# Define the boundaries of the plot
x_min, x_max = 0, 10
y_min, y_max = 0, 10

# Create a grid of points
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 400),
                             np.linspace(y_min, y_max, 400))

# Define the linear inequality constraints
# These define the half-planes
constraints = [
    (x_grid >= 2),            # x >= 2 (Vertical line)
    (y_grid >= 1.5),          # y >= 1.5 (Horizontal line)
    (x_grid <= 8),            # x <= 8 (Vertical line)
    (y_grid <= 9),            # y <= 9 (Horizontal line)
    (y_grid <= -1.0 * x_grid + 12), # y <= -x + 12 (Diagonal line)
    (y_grid >= 0.5 * x_grid + 0)   # y >= 0.5x     (Diagonal line)
]

# Combine constraints: find where *all* are true
feasible = np.logical_and.reduce(constraints)

# Create the plot
plt.figure(figsize=(7, 7))

# Plot the boundary lines for clarity
x_vals = np.linspace(x_min, x_max, 100)
plt.plot(np.ones_like(x_vals)*2, x_vals, 'k--', label='x = 2') # Used x_vals for y range here
plt.plot(x_vals, np.ones_like(x_vals)*1.5, 'k--', label='y = 1.5')
plt.plot(np.ones_like(x_vals)*8, x_vals, 'k--', label='x = 8') # Used x_vals for y range here
plt.plot(x_vals, np.ones_like(x_vals)*9, 'k--', label='y = 9')
plt.plot(x_vals, -1.0 * x_vals + 12, 'k--', label='y = -x + 12')
plt.plot(x_vals, 0.5 * x_vals + 0, 'k--', label='y = 0.5x')

# Shade the feasible region
# Use imshow to show the boolean mask, setting origin and extent
plt.imshow(feasible, extent=(x_min, x_max, y_min, y_max), origin='lower',
           cmap='Greens', alpha=0.4)

# Find contours to outline the feasible region more sharply (optional but nice)
plt.contour(x_grid, y_grid, feasible, levels=[0.5], colors='green', linewidths=2)


# Add annotations (arrows indicating allowed side)
plt.annotate('', xy=(5, 6), xytext=(4, 6), arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate('', xy=(6, 3), xytext=(6, 4), arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate('', xy=(3, 7), xytext=(3, 8), arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate('', xy=(7, 3), xytext=(8, 3), arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate('', xy=(5, 6), xytext=(6, 7), arrowprops=dict(arrowstyle='->', color='blue')) # For y <= -x + 12
plt.annotate('', xy=(4, 3), xytext=(3, 2), arrowprops=dict(arrowstyle='->', color='blue')) # For y >= 0.5x

plt.text(4.5, 4.5, 'Feasible Region\n(Convex Polyhedron)', ha='center', va='center', fontsize=12, color='darkgreen', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))


# Setup plot appearance
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Variable $x_1$ (e.g., component of g, u, y, ...)')
plt.ylabel('Variable $x_2$ (e.g., component of g, u, y, ...)')
plt.title('Intersection of Linear Constraints Forms a Convex Feasible Set')
plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(loc='upper right') # Legend can get cluttered, removed for clarity
plt.gca().set_aspect('equal', adjustable='box') # Make aspect ratio equal

# Save the figure
# plt.savefig('feasible_region_polyhedron.png', dpi=300)
plt.show() # Display the plot
