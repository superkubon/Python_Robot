from matplotlib import pyplot as plt
import re

# Load the uploaded G-code file
gcode_path = "D:/Work/Thesis/Robot_python/input_gcode/6_gcode.nc"

# Initialize lists to store X and Y coordinates
x_vals = []
y_vals = []

# Regular expression to find G-code lines with X and Y coordinates
pattern = re.compile(r'X(-?/d+/.?/d*)/s*Y(-?/d+/.?/d*)')

# Read and parse the G-code file
with open(gcode_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            x_vals.append(x)
            y_vals.append(y)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', label='G-code Path')
plt.title("Visualization of G-code Path")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()
