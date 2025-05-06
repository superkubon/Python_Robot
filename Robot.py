import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH
from scipy.spatial import Delaunay, ConvexHull

# Define robot
L = [
    RevoluteDH(d=0.1687, a=0, alpha=np.pi/2),      # Joint 1
    RevoluteDH(d=0, a=0.1556, alpha=0),             # Joint 2
    RevoluteDH(d=0.2271, a=0, alpha=-np.pi/2),      # Joint 3
    RevoluteDH(d=0.16, a=0, alpha=np.pi/2),         # Joint 4
    RevoluteDH(d=0, a=0.16, alpha=np.pi/2)          # Joint 5
]

robot = DHRobot(L, name='My5DOFRobot')

# Function to convert theta (deg) to motor steps and direction
def convert_theta_to_steps_with_dir(theta_list_deg):
    motor_resolution_deg = [0.45, 0.225, 0.45, 0.0140625, 0.225]  # degrees per step
    steps_with_dir = []
    for theta, res in zip(theta_list_deg, motor_resolution_deg):
        steps = abs(round(theta / res))
        direction = 1 if theta >= 0 else 0  # 1 = forward, 0 = reverse
        steps_with_dir.append((steps, direction))
    return steps_with_dir

# Function to apply backlash compensation
def apply_backlash_compensation(steps_dir_list, prev_dirs, backlash_steps=[2, 4, 2, 10, 4]):
    corrected = []
    updated_prev_dirs = prev_dirs.copy()
    for i, (steps, dir_current) in enumerate(steps_dir_list):
        steps_corrected = steps
        if dir_current != prev_dirs[i]:
            steps_corrected += backlash_steps[i]
            print(f'Joint {i+1}: Direction changed, adding {backlash_steps[i]} steps for backlash compensation.')
        corrected.append((steps_corrected, dir_current))
        updated_prev_dirs[i] = dir_current
    return corrected, updated_prev_dirs

# Initialize home configuration
q = np.array([0, 0, 0, 0, 0])

# Plot the robot
robot.plot(q, limits=[-1, 1, -1, 1, 0, 1])

# Teach pendant
robot.teach(q)

# FK example
T = robot.fkine(q)
print('T =\n', T)

# IK example
sol = robot.ikine_LM(T, mask=[1,1,1,0,0,1])
print('Inverse kinematics solution (deg):', np.degrees(sol.q))

# Sample random points for workspace visualization
num_points = 3000
q_min = np.radians([-180, -90, -90, -90, -180])
q_max = np.radians([180, 90, 90, 90, 180])

points = []

for _ in range(num_points):
    q_rand = q_min + (q_max - q_min) * np.random.rand(robot.n)
    T_rand = robot.fkine(q_rand)
    points.append(T_rand.t)

points = np.array(points)

# Build a 3D boundary
hull = ConvexHull(points)

# Plot workspace
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=hull.simplices, color='cyan', alpha=0.3)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Robot 3D Workspace')
ax.set_box_aspect([1,1,1])
plt.show()

# IK Test Cases
T_list = [
    SE3(0.4, 0, 0.4),
    SE3(0.7, 0, 0.2),
    SE3(0.3, 0.3, 0.4),
    SE3(0, 0, 0.7),
    SE3(0.5, 0.5, 0.5),
    SE3(-0.25, -0.30, 0.46)
]

position_error = []
prev_dirs = [1, 1, 1, 1, 1]  # assume all joints start with forward direction

for i, T_target in enumerate(T_list):
    sol = robot.ikine_LM(T_target, mask=[1,1,1,0,0,0])
    
    if not sol.success:
        print(f'IK failed for test case {i+1}')
        position_error.append(np.nan)
        continue
    
    q_sol = sol.q
    T_check = robot.fkine(q_sol)
    
    error = np.linalg.norm(T_target.t - T_check.t)
    position_error.append(error)
    
    print(f'\nTest case {i+1}:')
    print('IK Solution (deg):', np.degrees(q_sol))
    print(f'Position error = {error:.6f} meters')

    # Convert theta to motor steps and direction
    theta_deg = np.degrees(q_sol)
    motor_steps_and_dir = convert_theta_to_steps_with_dir(theta_deg)
    print('Motor steps and direction (before backlash compensation):', motor_steps_and_dir)

    # Apply backlash compensation
    motor_steps_with_backlash, prev_dirs = apply_backlash_compensation(motor_steps_and_dir, prev_dirs)
    print('Motor steps and direction (after backlash compensation):', motor_steps_with_backlash)

# Summary
valid_errors = [e for e in position_error if not np.isnan(e)]
print('\nAverage position error (excluding failed cases):', np.mean(valid_errors))
