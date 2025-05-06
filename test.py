import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
import csv
from roboticstoolbox import DHRobot, RevoluteDH
from scipy.spatial import ConvexHull
from spatialmath.base import tr2eul

# --------------------- Robot Definition ---------------------
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
feedrates = [2700, 3300, 3000, 3500, 3200]  # mm/min

# --------------------- Utility Functions ---------------------
def convert_theta_to_steps_with_dir(theta_list_deg, gear_ratios=None):
    if gear_ratios is None:
        gear_ratios = [1.0] * 5
    steps_with_dir = []
    for theta, res, gear in zip(theta_list_deg, motor_resolution_deg, gear_ratios):
        effective_theta = theta / gear
        steps = abs(round(effective_theta / res))
        direction = 1 if theta >= 0 else 0
        steps_with_dir.append((steps, direction))
    return steps_with_dir

from math import ceil

def apply_backlash_compensation(steps_dir_list, prev_dirs):
    # Tính số bước cần bù dựa trên thông số backlash thực tế
    backlash_degrees = [0.1167, 0.05, 0, 0, 0]  # giả định khớp 3–5 nếu chưa có dữ liệu
    # motor_resolution_deg = [0.225, 0.45, 0.45, 0.0140625, 0.225]
    backlash_steps = [ceil(b / res) for b, res in zip(backlash_degrees, motor_resolution_deg)]

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


def make_target_with_z_up(x, y, z):
    z_axis = np.array([0, 0, 1], dtype=float)
    x_temp = np.array([1, 0, 0], dtype=float)
    y_axis = np.cross(z_axis, x_temp)
    if np.linalg.norm(y_axis) == 0:
        x_temp = np.array([0, 1, 0], dtype=float)
        y_axis = np.cross(z_axis, x_temp)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))
    return SE3.Rt(R, [x, y, z])

def check_z_alignment(q_sol):
    T4 = robot.fkine(q_sol, end=3)
    T5 = robot.fkine(q_sol)
    z4 = T4.R[:,2]
    z5 = T5.R[:,2]
    dot_product = np.dot(z4, z5)
    alignment_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    # if alignment_deg > 5:
    #     print(f'⚠️  Cảnh báo: trục Z của khớp 4 và khâu cuối lệch {alignment_deg:.2f}°')
    # else:
    #     print(f'✅ Trục Z của khớp 4 và end-effector lệch {alignment_deg:.2f}° (chấp nhận được)')

def format_motion_mapping(motion_mm):
    labels = ['X', 'Y', 'Z', 'A', 'B']
    return ' '.join(f'{label}{val:.2f}' for label, val in zip(labels, motion_mm))

def generate_gcode(motion_mm):
    labels = ['X', 'Y', 'Z', 'A', 'B']
    t_i = [motion / fr for motion, fr in zip(motion_mm, feedrates)]
    t_max = max(t_i)
    synced_feedrates = [motion / t_max for motion in motion_mm]
    gcode_motion = ' '.join(f'{label}{motion:.2f}' for label, motion in zip(labels, motion_mm))
    gcode_feed = f'F{round(max(synced_feedrates))}'
    print(f'🧾 G-code: G1 {gcode_motion} {gcode_feed}')
    print(f'⏱  Thời gian hoàn thành: {t_max*60:.2f} giây')

def check_ik_accuracy(T_target, T_check):
    position_error = np.linalg.norm(T_target.t - T_check.t)
    orientation_error = np.linalg.norm(tr2eul(T_target.R) - tr2eul(T_check.R))
    print(f"📍 Sai số vị trí (m): {position_error:.6f}")
    # print(f"📐 Sai số định hướng (rad): {orientation_error:.6f}")
    return position_error, orientation_error

def check_step_conversion_accuracy(theta_deg, steps_with_backlash):
    theta_recovered = [step * res / gear for (step, _), res, gear in zip(steps_with_backlash, motor_resolution_deg, gear_ratios)]
    recovery_error = np.abs(np.array(theta_deg) - np.array(theta_recovered))
    print("🔁 Sai số quy đổi góc (deg):", recovery_error.round(4))
    return recovery_error

def run_workspace_visualization():
    num_points = 3000
    q_min = np.radians([-180, -90, -90, -90, -180])
    q_max = np.radians([180, 90, 90, 90, 180])
    points = []

    for _ in range(num_points):
        q_rand = q_min + (q_max - q_min) * np.random.rand(robot.n)
        T_rand = robot.fkine(q_rand)
        points.append(T_rand.t)

    points = np.array(points)
    hull = ConvexHull(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=hull.simplices, color='cyan', alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot 3D Workspace')
    ax.set_box_aspect([1,1,1])
    plt.show()

def interactive_prompt():
    print("\\n🔧 Nhập vị trí mong muốn (x, y, z) và chọn hướng khâu cuối:")
    try:
        x = float(input("Nhập x (m): "))
        y = float(input("Nhập y (m): "))
        z = float(input("Nhập z (m): "))
        upright = input("Khâu cuối phải thẳng đứng? (y/n): ").strip().lower() == 'y'
        T_target = make_target_with_z_up(x, y, z) if upright else SE3(x, y, z)
        mask = [1,1,1,0,0,1] if upright else [1,1,1,0,0,0]
        sol = robot.ikine_LM(T_target, mask=mask)
        if not sol.success:
            print("❌ Không tìm được nghiệm IK.")
            return
        q_sol = sol.q
        T_check = robot.fkine(q_sol)
        check_ik_accuracy(T_target, T_check)
        theta_deg = np.degrees(q_sol)
        motor_steps_and_dir = convert_theta_to_steps_with_dir(theta_deg, gear_ratios)
        print('Steps + Direction (trước backlash compensation):', motor_steps_and_dir)
        motor_steps_with_backlash, _ = apply_backlash_compensation(motor_steps_and_dir, [1,1,1,1,1])
        print('Steps + Direction (sau backlash compensation):', motor_steps_with_backlash)
        _ = check_step_conversion_accuracy(theta_deg, motor_steps_with_backlash)
        motion_mm = [steps / 500.0 for steps, _ in motor_steps_with_backlash]
        print('Mapping:', format_motion_mapping(motion_mm))
        generate_gcode(motion_mm)
        check_z_alignment(q_sol)
        robot.plot(q_sol, block=False)
        plt.title("Robot Configuration - Manual Input")
        plt.show()
    except ValueError:
        print("⚠️ Lỗi: Giá trị nhập không hợp lệ.")

import csv

def run_test_cases():
    T_list = [
        make_target_with_z_up(0.4, 0, 0.4),
        make_target_with_z_up(0.7, 0, 0.2),
        make_target_with_z_up(0.3, 0.3, 0.4),
        make_target_with_z_up(0, 0, 0.7),
        make_target_with_z_up(0.5, 0.5, 0.5),
        make_target_with_z_up(-0.25, -0.30, 0.46)
    ]

    position_errors = []
    orientation_errors = []
    prev_dirs = [1, 1, 1, 1, 1]
    microsteps_all_cases = []

    for i, T_target in enumerate(T_list):
        sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 1])
        if not sol.success:
            print(f'❌ IK failed for test case {i+1}')
            position_errors.append(np.nan)
            orientation_errors.append(np.nan)
            continue

        q_sol = sol.q
        T_check = robot.fkine(q_sol)
        print(f'\n🧪 Test case {i+1}:')
        print('Góc khớp (deg):', np.degrees(q_sol))

        pos_err, orient_err = check_ik_accuracy(T_target, T_check)
        position_errors.append(pos_err)
        orientation_errors.append(orient_err)

        theta_deg = np.degrees(q_sol)
        motor_steps_and_dir = convert_theta_to_steps_with_dir(theta_deg, gear_ratios)
        print('Steps + Direction (trước backlash compensation):', motor_steps_and_dir)

        motor_steps_with_backlash, prev_dirs = apply_backlash_compensation(motor_steps_and_dir, prev_dirs)
        print('Steps + Direction (sau backlash compensation):', motor_steps_with_backlash)

        _ = check_step_conversion_accuracy(theta_deg, motor_steps_with_backlash)

        motion_mm = [steps / 500.0 for steps, _ in motor_steps_with_backlash]
        print('Mapping:', format_motion_mapping(motion_mm))

        generate_gcode(motion_mm)
        check_z_alignment(q_sol)

        robot.plot(q_sol, block=False)
        plt.title(f'Test Case {i+1}')
        plt.show()

        # --- Ghi lại microstep ---
        microstep_dict = {
            "Test case": i+1,
            "X": motor_steps_with_backlash[0][0],
            "Y": motor_steps_with_backlash[1][0],
            "Z": motor_steps_with_backlash[2][0],
            "A": motor_steps_with_backlash[3][0],
            "B": motor_steps_with_backlash[4][0]
        }
        microsteps_all_cases.append(microstep_dict)

    # ✅ Hiển thị kết quả microsteps
    print('\n📄 Tổng số microsteps cho từng test case:')
    for entry in microsteps_all_cases:
        print(entry)

    # ✅ Ghi vào file CSV
    with open("microsteps_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Test case", "X", "Y", "Z", "A", "B"])
        writer.writeheader()
        writer.writerows(microsteps_all_cases)

    # ✅ Trung bình sai số
    valid_pos = [e for e in position_errors if not np.isnan(e)]
    valid_orient = [e for e in orientation_errors if not np.isnan(e)]
    print('\n📊 Trung bình sai số vị trí (m):', np.mean(valid_pos))
    print('📊 Trung bình sai số định hướng (rad):', np.mean(valid_orient))


if __name__ == "__main__":
    while True:
        print("\\n==================== Menu ====================")
        print("1. Hiển thị không gian làm việc")
        print("2. Chạy các test case")
        print("3. Nhập vị trí mong muốn")
        print("4. Thoát")
        choice = input("Chọn tùy chọn (1-4): ")
        if choice == '1':
            run_workspace_visualization()
        elif choice == '2':
            run_test_cases()
        elif choice == '3':
            interactive_prompt()
        elif choice == '4':
            break
        else:
            print("⚠️ Lỗi: Vui lòng chọn tùy chọn hợp lệ.")