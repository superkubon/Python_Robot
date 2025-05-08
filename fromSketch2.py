import numpy as np
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH
from math import degrees
import time

# === Robot Definition ===
L = [
    RevoluteDH(d=0.1687, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0.1556, alpha=0),
    RevoluteDH(d=0.2271, a=0, alpha=-np.pi/2),
    RevoluteDH(d=0.16, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0.16, alpha=np.pi/2)
]
robot = DHRobot(L, name='My5DOFRobot')

# === Motor Parameters ===
gear_ratios = [1/40, 1/81, 1.0, 1.0, 1/36]
motor_resolution_deg = [0.225, 0.45, 0.45, 0.0140625, 0.225]
backlash_degrees = [0.1167, 0.05, 0, 0, 0]

# === Orientation Candidates ===
ORIENTATION_LIST = [
    SE3.Ry(np.pi),                        # Z down
    SE3(),                                # No rotation
    SE3.Ry(np.pi/2),                      # Front
    SE3.Rx(np.pi/2),                      # Left
    SE3.Rz(np.pi/2) * SE3.Ry(np.pi),      # Diagonal Z-down
    SE3.Ry(np.pi/4),                      # Slight tilt
]

# === Load and scale G-code coordinates ===
def load_gcode(filename, scale=10.0, default_z=100.0):
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('G1') or line.startswith('G0'):
                parts = line.strip().split()
                x = y = z = None
                for p in parts:
                    if p.startswith('X'): x = float(p[1:]) * scale
                    if p.startswith('Y'): y = float(p[1:]) * scale
                    if p.startswith('Z'): z = float(p[1:]) * scale
                if x is not None and y is not None:
                    if z is None: z = default_z
                    yield (x/1000, y/1000, z/1000)  # convert to meters

# === Try multiple orientations ===
def try_multiple_orientations(x, y, z, robot):
    for orient in ORIENTATION_LIST:
        T = SE3(x, y, z) * orient
        ik = robot.ikine_LM(T)
        if ik.success:
            return ik, orient
    return None, None

# === Convert joint angles to steps ===
def convert_theta_to_steps(theta_deg, motor_res, gear_ratios):
    steps = []
    for angle, res, ratio in zip(theta_deg, motor_res, gear_ratios):
        step = round(angle / (res * ratio))
        steps.append(step)
    return steps


# === Main routine ===
def main():
    input_file = "D:/Work/Thesis/Robot_python/input_gcode/6_gcode.txt"
    gcode_lines = []

    for x, y, z in load_gcode(input_file, scale=10.0, default_z=100.0):
        ik, used_orient = try_multiple_orientations(x, y, z, robot)
        if ik is None:
            print(f"❌ IK failed at ({x:.4f}, {y:.4f}, {z:.4f})")
            continue

        angles_deg = [degrees(a) for a in ik.q]
        steps = convert_theta_to_steps(angles_deg, motor_resolution_deg, gear_ratios)
        g_line = f"G1 X{steps[0]} Y{steps[1]} Z{steps[2]} A{steps[3]} B{steps[4]} F2700"
        gcode_lines.append(g_line)

    if not gcode_lines:
        print("⚠️ No valid IK results. Nothing to send.")
        return


if __name__ == "__main__":
    main()
