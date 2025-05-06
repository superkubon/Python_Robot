import numpy as np
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH
from math import ceil

L = [
    RevoluteDH(d=0.1687, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0.1556, alpha=0),
    RevoluteDH(d=0.2271, a=0, alpha=-np.pi/2),
    RevoluteDH(d=0.16, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0.16, alpha=np.pi/2)
]
robot = DHRobot(L, name='My5DOFRobot')

gear_ratios = [1/40, 1/81, 1.0, 1.0, 1/36]
motor_resolution_deg = [0.225, 0.45, 0.45, 0.0140625, 0.225]
Z_DRAWING_PLANE = 100  # mm
SCALE_FACTOR = 10.0  # adjustable

def parse_gcode_line(line):
    line = line.strip()
    if line.startswith("G0") or line.startswith("G1"):
        parts = line.split()
        if len(parts) >= 3 and parts[1].startswith("X") and parts[2].startswith("Y"):
            x = float(parts[1][1:])
            y = float(parts[2][1:])
            return x, y
    return None

def make_target_position_only(x, y, z):
    return SE3(x / 1000.0, y / 1000.0, z / 1000.0)

def convert_theta_to_steps_with_dir(theta_list_deg):
    steps_with_dir = []
    for theta, res, gear in zip(theta_list_deg, motor_resolution_deg, gear_ratios):
        effective_theta = theta / gear
        steps = round(effective_theta / res)
        direction = 1 if steps >= 0 else 0
        steps_with_dir.append((abs(steps), direction))
    return steps_with_dir

def apply_backlash_compensation(steps_dir_list, prev_dirs):
    backlash_degrees = [0.1167, 0.05, 0, 0, 0]
    backlash_steps = [ceil(b / res) for b, res in zip(backlash_degrees, motor_resolution_deg)]
    corrected = []
    updated_prev_dirs = prev_dirs.copy()
    for i, (steps, dir_current) in enumerate(steps_dir_list):
        steps_corrected = steps
        if dir_current != prev_dirs[i]:
            steps_corrected += backlash_steps[i]
        corrected.append((steps_corrected, dir_current))
        updated_prev_dirs[i] = dir_current
    return corrected, updated_prev_dirs

def convert_gcode_to_motor_movement(input_path, output_path):
    prev_dirs = [1] * 5
    prev_steps = [0] * 5
    converted_gcode = []
    total_lines = 0
    successful = 0

    with open(input_path, "r") as f:
        for line in f:
            total_lines += 1
            result = parse_gcode_line(line)
            if not result:
                print(f"⏭️  Skipping line {total_lines}: {line.strip()}")
                continue
            x, y = result
            x *= SCALE_FACTOR
            y *= SCALE_FACTOR
            T_target = make_target_position_only(x, y, Z_DRAWING_PLANE)
            sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])
            if not sol.success:
                print(f"⚠️  IK failed at line {total_lines} → point ({x:.2f}, {y:.2f})")
                continue
            theta_deg = np.degrees(sol.q)
            steps_with_dir = convert_theta_to_steps_with_dir(theta_deg)
            steps_with_backlash, prev_dirs = apply_backlash_compensation(steps_with_dir, prev_dirs)

            # Compute relative movement (Δstep → Δmm)
            current_steps = [s for s, _ in steps_with_backlash]
            delta_steps = [abs(c - p) for c, p in zip(current_steps, prev_steps)]
            motion_mm = [ds / 500.0 for ds in delta_steps]
            prev_steps = current_steps

            converted_gcode.append(
                f"G1 X{motion_mm[0]:.2f} Y{motion_mm[1]:.2f} Z{motion_mm[2]:.2f} "
                f"A{motion_mm[3]:.2f} B{motion_mm[4]:.2f} F2700"
            )
            successful += 1

    with open(output_path, "w") as fout:
        fout.write("\n".join(converted_gcode))
    print(f"✅ Generated {successful}/{total_lines} G-code lines. Saved to {output_path}")

if __name__ == "__main__":
    convert_gcode_to_motor_movement("D:/Work/Thesis/Robot_python/input_gcode/6_gcode.nc", "gcode_motor_movement_relative.txt")