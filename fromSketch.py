import numpy as np
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH
from math import ceil
import matplotlib.pyplot as plt

# -------------------- Robot cấu hình --------------------
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
backlash_degrees = [0.1167, 0.05, 0, 0, 0]
Z_DRAWING_PLANE = 300  # mm
offset_x = 150  # mm
offset_y = 150  # mm

# -------------------- G-code parsing --------------------
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

# -------------------- Chuyển góc sang bước động cơ --------------------
def convert_theta_to_steps_with_dir(theta_list_deg):
    steps_with_dir = []
    for theta, res, gear in zip(theta_list_deg, motor_resolution_deg, gear_ratios):
        effective_theta = theta / gear
        steps = round(effective_theta / res)
        direction = 1 if steps >= 0 else 0
        steps_with_dir.append((abs(steps), direction))
    return steps_with_dir

def apply_backlash_compensation(steps_dir_list, prev_dirs):
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

# -------------------- Chuyển G-code sang bước động cơ --------------------
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
            x += offset_x
            y += offset_y
            T_target = make_target_position_only(x, y, Z_DRAWING_PLANE)
            sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])
            if not sol.success:
                print(f"⚠️  IK failed at line {total_lines} → point ({x:.2f}, {y:.2f})")
                continue
            theta_deg = np.degrees(sol.q)
            steps_with_dir = convert_theta_to_steps_with_dir(theta_deg)
            steps_with_backlash, prev_dirs = apply_backlash_compensation(steps_with_dir, prev_dirs)

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

# -------------------- Vẽ đường đi đầu vào --------------------
def visualize_input_path(input_path):
    x_vals = []
    y_vals = []

    with open(input_path, "r") as f:
        for line in f:
            result = parse_gcode_line(line)
            if not result:
                continue
            x, y = result
            x_vals.append(x + offset_x)
            y_vals.append(y + offset_y)

    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue')
    plt.title("Offset G-code Drawing Path (in mm)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# -------------------- Kiểm tra IK-FK --------------------
def verify_ik_with_fk(input_path):
    total = 0
    passed = 0
    tolerance = 0.05  # mm

    with open(input_path, "r") as f:
        for line in f:
            result = parse_gcode_line(line)
            if not result:
                continue

            x, y = result
            x += offset_x
            y += offset_y
            z = Z_DRAWING_PLANE
            T_target = make_target_position_only(x, y, z)
            sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])
            if not sol.success:
                print(f"⚠️ IK failed at ({x:.3f}, {y:.3f}, {z:.3f})")
                continue

            T_check = robot.fkine(sol.q)
            pos_fk = T_check.t * 1000  # m → mm
            error = np.linalg.norm(pos_fk - np.array([x, y, z]))

            if error < tolerance:
                passed += 1
            else:
                print(f"❌ Line {total+1}: Error = {error:.4f} mm")
                print(f"   Input  = ({x:.2f}, {y:.2f}, {z:.2f})")
                print(f"   FK out = ({pos_fk[0]:.2f}, {pos_fk[1]:.2f}, {pos_fk[2]:.2f})")

            total += 1

    print(f"✅ IK → FK verified: {passed}/{total} points passed within {tolerance:.2f} mm tolerance.")
def visualize_high_error_points(input_path, error_threshold=0.05):
    x_all, y_all = [], []
    x_err, y_err = [], []

    with open(input_path, "r") as f:
        for idx, line in enumerate(f):
            result = parse_gcode_line(line)
            if not result:
                continue

            x, y = result
            x += offset_x
            y += offset_y
            z = Z_DRAWING_PLANE
            T_target = make_target_position_only(x, y, z)
            sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])
            if not sol.success:
                continue

            T_fk = robot.fkine(sol.q)
            pos_fk = T_fk.t * 1000
            error = np.linalg.norm(pos_fk - np.array([x, y, z]))

            x_all.append(x)
            y_all.append(y)
            if error > error_threshold:
                x_err.append(x)
                y_err.append(y)

    plt.figure(figsize=(8, 8))
    plt.plot(x_all, y_all, linestyle='-', color='lightgray', label='All path')
    plt.scatter(x_err, y_err, color='red', label=f'Error > {error_threshold} mm', zorder=10)
    plt.title("High Error Points on XY Drawing Path")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def visualize_fk_drawing_from_gcode(input_path):
    x_fk = []
    y_fk = []

    with open(input_path, "r") as f:
        for idx, line in enumerate(f):
            result = parse_gcode_line(line)
            if not result:
                continue

            x, y = result
            x += offset_x
            y += offset_y
            z = Z_DRAWING_PLANE
            T_target = make_target_position_only(x, y, z)

            sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])
            if not sol.success:
                print(f"⚠️ IK failed at line {idx + 1}")
                continue

            T_fk = robot.fkine(sol.q)
            pos_mm = T_fk.t * 1000  # Convert m to mm

            x_fk.append(pos_mm[0])
            y_fk.append(pos_mm[1])

    plt.figure(figsize=(8, 8))
    plt.plot(x_fk, y_fk, marker='o', linestyle='-', color='green')
    plt.title("Sketch Reconstructed from Forward Kinematics")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show(block=False)


def visualize_first_n_points_fk(input_path, n=4):
    x_input, y_input = [], []
    x_fk, y_fk = [], []

    with open(input_path, "r") as f:
        count = 0
        for idx, line in enumerate(f):
            if count >= n:
                break

            result = parse_gcode_line(line)
            if not result:
                continue

            x, y = result
            x_offset = x + offset_x
            y_offset = y + offset_y
            T_target = make_target_position_only(x_offset, y_offset, Z_DRAWING_PLANE)

            sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])
            if not sol.success:
                print(f"⚠️ IK failed at line {idx+1}")
                continue

            T_fk = robot.fkine(sol.q)
            pos_fk = T_fk.t * 1000  # m → mm

            x_input.append(x_offset)
            y_input.append(y_offset)
            x_fk.append(pos_fk[0])
            y_fk.append(pos_fk[1])
            count += 1

    plt.figure(figsize=(8, 8))
    plt.plot(x_input, y_input, 'bo-', label='Original G-code Points (Offset)')
    plt.plot(x_fk, y_fk, 'go--', label='FK Positions after IK')
    for i in range(len(x_input)):
        plt.text(x_input[i], y_input[i], f'P{i+1}', fontsize=10, color='blue')
        plt.text(x_fk[i], y_fk[i], f'FK{i+1}', fontsize=10, color='green')
    plt.title("First N Sketch Points: G-code vs FK")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# -------------------- Chạy toàn bộ --------------------
if __name__ == "__main__":
    input_gcode = "D:/Work/Thesis/Robot_python/input_gcode/6_gcode.nc"
    output_gcode = "D:/Work/Thesis/Robot_python/output_gcode/gcode_motor_movement_relative.txt"
    
    convert_gcode_to_motor_movement(input_gcode, output_gcode)
    # visualize_input_path(input_gcode)
    verify_ik_with_fk(input_gcode)
    visualize_high_error_points(input_gcode)
    visualize_fk_drawing_from_gcode(input_gcode)
    visualize_first_n_points_fk(input_gcode, n=400)



