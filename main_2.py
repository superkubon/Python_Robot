import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import re

# ===== Định nghĩa robot =====
L1, L2, L3 = 0.2, 0.15, 0.18
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=0, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0, offset=-np.pi/3, qlim=[-75 * deg, 75 * deg]),
    RevoluteDH(d=0, a=L3, alpha=0, offset=np.pi/2, qlim=[-120 * deg, 120 * deg])
], name='3DOF_Robot')
q = np.zeros(3)  # Khởi tạo nghiệm ban đầu là 0
robot.teach(q)
# ===== Hệ số chuyển đổi góc → bước (mm) =====
STEP_CONVERT = {
    'X': 0.355555,
    'Y': 0.355555,
    'Z': 0.216666
}
def interpolate_cartesian(p_start, p_end, step_size=0.01):
    """
    Nội suy tuyến tính giữa hai điểm trong không gian 3D với khoảng cách đều nhau.
    """
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    distance = np.linalg.norm(p_end - p_start)
    if distance == 0:
        return [p_start.tolist()]
    num_steps = int(distance / step_size)
    return [(p_start + (i / num_steps) * (p_end - p_start)).tolist() for i in range(1, num_steps + 1)]

# ===== Hàm tính G-code giữ G0/G1 và ưu tiên nghiệm gần nhất =====
def compute_gcode_line(cmd, x, y, z, q0=None, max_attempts=10):
    T_goal = SE3(x, y, z)

    for attempt in range(max_attempts):
        ik_result = robot.ikine_LM(T_goal, q0=q0, mask=[1, 1, 1, 0, 0, 0])
        if not ik_result.success:
            continue

        q_deg = np.degrees(ik_result.q)

        if -90 < q_deg[0] < 90 and -80 < q_deg[1] < 80 and -80 < q_deg[2] < 80:
            x_step = -q_deg[0] * STEP_CONVERT['X']
            y_step = -q_deg[1] * STEP_CONVERT['Y'] - 20
            z_step = q_deg[2] * STEP_CONVERT['Z'] + 20
            gcode_line = f"{cmd} X{x_step:.3f} Y{y_step:.3f} Z{z_step:.3f} F2700"
            return gcode_line, q_deg, ik_result.q

    return None, None, None

# ===== Đọc và xử lý file G-code =====
input_file = "D:/Work/Thesis/Robot_python/input_gcode/square_gcode_2.nc"
pattern = re.compile(r"^(G0|G1)\s+.*?X([-+]?\d*\.?\d+)\s+Y([-+]?\d*\.?\d+)(?:\s+Z([-+]?\d*\.?\d+))?", re.IGNORECASE)

gcode_lines = []
q_list = []
q0 = None  # Khởi tạo nghiệm ban đầu là None
def set_axes_equal(ax):
    import numpy as np

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


last_point = None
SCALE = 1
y_fixed = 0.2  # Y cố định
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        match = pattern.search(line)
        if not match:
            if line.startswith(("M3", "M5", "G28")):
                gcode_lines.append(line)
            continue

        cmd = match.group(1).upper()
        x_gcode = float(match.group(2))
        y_gcode = float(match.group(3))
        z_gcode = float(match.group(4)) if match.group(4) is not None else 0.0

        # Chuyển đổi sang đơn vị mét
        x = (x_gcode / 1000.0) * SCALE
        y = y_fixed
        z = (y_gcode / 1000.0) * SCALE + 0.2

        current_point = [x, y, z]

        if last_point is not None:
            interp_points = interpolate_cartesian(last_point, current_point, step_size=0.001)
        else:
            interp_points = [current_point]

        for pt in interp_points:
            gcode_line, q_deg, q_rad = compute_gcode_line(cmd, pt[0], pt[1], pt[2], q0=q0)
            if q_rad is None:
                print(f"❌ IK thất bại tại điểm ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
            else:
                gcode_lines.append(gcode_line)
                q_list.append(q_deg)
                q0 = q_rad

        last_point = current_point

# ===== Ghi file kết quả G-code =====
output_file = "square3.txt"
with open(output_file, "w") as f:
    for line in gcode_lines:
        f.write(line + "\n")
print(f"\n✅ Đã lưu {len(gcode_lines)} dòng vào '{output_file}'")

# ===== Ghi file góc khớp ra file riêng =====
angle_file = "square_joint_angles3.txt"
with open(angle_file, "w") as f:
    f.write("q1_deg,q2_deg,q3_deg\n")
    for q_deg in q_list:
        f.write("{:.4f},{:.4f},{:.4f}\n".format(*q_deg))
print(f"✅ Đã lưu góc khớp vào '{angle_file}'")

#===== Plot đường đi đầu cuối =====
if q_list:
    positions = [robot.fkine(np.radians(q)).t for q in q_list]
    xs = [p[0] for p in positions]  # X thật
    ys = [p[1] for p in positions]  # Z thật gán cho Y plot
    zs = [p[2] for p in positions]  # Y thật gán cho Z plot

    for i, pos in enumerate(positions):
        print(f"Điểm {i}: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        print(f"Plot point {i}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, zs, ys, marker='o', label='Quỹ đạo đầu cuối')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")  # đổi nhãn ở đây
    ax.set_zlabel("Y (m)")  # đổi nhãn ở đây
    print("Giá trị ys (trục Z plot):", ys)
    ax.set_title("Đường đi thực tế của đầu cuối robot")
    ax.legend()
    ax.view_init(elev=45, azim=45)
    set_axes_equal(ax)
    plt.show()

else:
    print("⚠️ Không có điểm nào để vẽ quỹ đạo.")
