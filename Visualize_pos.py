import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

# ===== Định nghĩa robot =====
L1, L2, L3, L4 = 0.1537, 0.1433, 0.0, 0.1443
laser_extension = 0.075  # 7.5 cm
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0, offset=-np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=L3, a=0, alpha=np.pi/2, offset=np.pi/2, qlim=[-120 * deg, 120 * deg]),
    RevoluteDH(d=L4, a=0, alpha=np.pi/2, offset=0, qlim=[-180 * deg, 180 * deg])
], name='3DOF_Robot')

q = np.zeros(4)  # Khởi tạo nghiệm ban đầu là 0
robot.teach(q)

STEP_CONVERT = {
    'X': 0.355555,
    'Y': 0.355555,
    'Z': 0.216666,
    'A': 0.133333,
}

def compute_gcode_line(cmd, x, y, z, q0=None, max_attempts=10):
    y_axis = np.array([0, -1, 0])  # Mục tiêu: Y luôn là [0, -1, 0]
    z_axis = np.array([0, 0, 1])
    x_axis = np.cross(y_axis, z_axis).astype(float)
    x_axis /= np.linalg.norm(x_axis)
    R_goal = np.column_stack((x_axis, y_axis, z_axis))
    T_goal = SE3(x, y, z)

    BACKLASH_X_DEG = 7 / 60
    BACKLASH_Y_DEG = 3 / 60
    prev_q_deg = None

    for attempt in range(max_attempts):
        ik_result = robot.ikine_LM(T_goal, q0=q0, mask=[1,1,1,0,0,0])
        if ik_result.success:
            q_deg = np.degrees(ik_result.q)
        else:
            continue
        # Bù backlash
        if prev_q_deg is not None:
            if np.sign(q_deg[0] - prev_q_deg[0]) != np.sign(prev_q_deg[0]):
                q_deg[0] += BACKLASH_X_DEG * np.sign(q_deg[0] - prev_q_deg[0])
            if np.sign(q_deg[1] - prev_q_deg[1]) != np.sign(prev_q_deg[1]):
                q_deg[1] += BACKLASH_Y_DEG * np.sign(q_deg[1] - prev_q_deg[1])
        prev_q_deg = q_deg.copy()
        if -90 < q_deg[0] < 90 and -120 < q_deg[1] < 120 and -120 < q_deg[2] < 120:
            x_step = -q_deg[0] * STEP_CONVERT['X']
            y_step = -q_deg[1] * STEP_CONVERT['Y']
            z_step = q_deg[2] * STEP_CONVERT['Z']
            a_step = q_deg[3] * STEP_CONVERT['A']
            gcode_line = f"{cmd} X{x_step:.3f} Y{y_step:.3f} Z{z_step:.3f} A{a_step:.3f} F2700"
            return gcode_line, q_deg, ik_result.q
    return None, None, None

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

# ===== Nhập tọa độ mục tiêu từ người dùng =====
try:
    x_input = float(input("Nhập tọa độ X mục tiêu (m): "))
    y_input = float(input("Nhập tọa độ Y mục tiêu (m): "))
    z_input = float(input("Nhập tọa độ Z mục tiêu (m): "))
except ValueError:
    print("❌ Giá trị nhập không hợp lệ. Sử dụng tọa độ mặc định (0.0, 0.2, 0.25).")
    x_input, y_input, z_input = 0.0, 0.2, 0.25

gcode_lines = []
q_list = []
q0 = None

# Di chuyển đến tâm mặc định
x_center_init, y_center_init, z_center_init = 0.0, 0.2, 0.25
init_line, q_deg_init, q_rad_init = compute_gcode_line("G1", x_center_init, y_center_init, z_center_init)
if init_line:
    print(f"🚀 Di chuyển đến tâm mặc định: {init_line}")
    print("🔧 Góc khớp (deg): q1 = {:.2f}, q2 = {:.2f}, q3 = {:.2f}, q4 = {:.2f}".format(*q_deg_init))
    gcode_lines.append(init_line)
    x_step = -q_deg_init[0] * STEP_CONVERT['X']
    y_step = -q_deg_init[1] * STEP_CONVERT['Y']
    z_step = q_deg_init[2] * STEP_CONVERT['Z']
    a_step = q_deg_init[3] * STEP_CONVERT['A']
    gcode_lines.append(f"G92 X{x_step:.3f} Y{y_step:.3f} Z{z_step:.3f} A{a_step:.3f}")
    q_list.append(q_deg_init)
    q0 = q_rad_init
else:
    print("❌ Không thể di chuyển đến tâm mặc định")

# Di chuyển đến vị trí nhập
target_line, q_deg_target, q_rad_target = compute_gcode_line("G1", x_input, y_input, z_input, q0=q0)
if target_line:
    print(f"🚀 Di chuyển đến vị trí nhập: {target_line}")
    print("🔧 Góc khớp (deg): q1 = {:.2f}, q2 = {:.2f}, q3 = {:.2f}, q4 = {:.2f}".format(*q_deg_target))
    gcode_lines.append(target_line)
    q_list.append(q_deg_target)
else:
    print(f"❌ IK thất bại tại vị trí nhập ({x_input}, {y_input}, {z_input})")

# Ghi file kết quả G-code
output_file = "D:/Work/Thesis/Robot_python/output_gcode/move_to_target.txt"
with open(output_file, "w") as f:
    for line in gcode_lines:
        f.write(line + "\n")
print(f"\n✅ Đã lưu {len(gcode_lines)} dòng G-code vào '{output_file}'")

# Ghi file góc khớp ra file riêng
angle_file = "D:/Work/Thesis/Robot_python/output_degree/move_to_target_angles.txt"
with open(angle_file, "w") as f:
    for q_deg in q_list:
        f.write("{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*q_deg))
print(f"✅ Đã lưu góc khớp vào '{angle_file}'")

# Vẽ quỹ đạo đầu cuối robot
if q_list:
    positions = [robot.fkine(np.radians(q)).t for q in q_list]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, zs, ys, marker='o', label='Quỹ đạo đầu cuối')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title("Đường đi thực tế của đầu cuối robot")
    ax.legend()
    ax.view_init(elev=45, azim=45)
    set_axes_equal(ax)

    # Vẽ khung 15x15cm ở vị trí cố định
    square_size = 0.1
    center_x = 0.0
    center_z = 0.15
    fixed_y = 0.2
    half = square_size / 2
    square_x = [center_x - half, center_x + half, center_x + half, center_x - half, center_x - half]
    square_z = [center_z - half, center_z - half, center_z + half, center_z + half, center_z - half]
    square_y = [fixed_y] * 5
    ax.plot(square_x, square_z, square_y, color='r', linestyle='--', label='Khung vẽ 15x15cm')

    plt.show()
else:
    print("⚠️ Không có điểm nào để vẽ quỹ đạo.")
