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

# ===== Hệ số chuyển đổi góc → bước (mm) =====
STEP_CONVERT = {
    'X': 0.355555,
    'Y': 0.355555,
    'Z': 0.216666
}

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
            y_step = -q_deg[1] * STEP_CONVERT['Y']
            z_step = q_deg[2] * STEP_CONVERT['Z']
            gcode_line = f"{cmd} X{x_step:.3f} Y{y_step:.3f} Z{z_step:.3f} F2700"
            return gcode_line, q_deg, ik_result.q

    return None, None, None

# ===== Đọc và xử lý file G-code =====
input_file = "D:/Work/Thesis/Robot_python/input_gcode/square_gcode.nc"
pattern = re.compile(r"^(G0|G1)\s+.*?X([-+]?\d*\.?\d+)\s+Y([-+]?\d*\.?\d+)(?:\s+Z([-+]?\d*\.?\d+))?", re.IGNORECASE)

gcode_lines = []
q_list = []
q0 = None  # Khởi tạo nghiệm ban đầu là None

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

        SCALE = 1
        x = (x_gcode / 1000.0) * SCALE
        y = 0.1
        z = (y_gcode / 1000.0) * SCALE

        gcode_line, q_deg, q_rad = compute_gcode_line(cmd, x, 0.1, z, q0=q0)
        if q_rad is None:
            print(f"❌ IK thất bại tại điểm ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"✅ {gcode_line}")
            print("🔧 Góc khớp (deg): q1 = {:.2f}, q2 = {:.2f}, q3 = {:.2f}".format(*q_deg))
            gcode_lines.append(gcode_line)
            q_list.append(q_deg)
            q0 = q_rad  # Cập nhật nghiệm cho bước sau

# ===== Ghi file kết quả G-code =====
output_file = "square2.txt"
with open(output_file, "w") as f:
    for line in gcode_lines:
        f.write(line + "\n")
print(f"\n✅ Đã lưu {len(gcode_lines)} dòng vào '{output_file}'")

# ===== Ghi file góc khớp ra file riêng =====
angle_file = "square_joint_angles2.txt"
with open(angle_file, "w") as f:
    f.write("q1_deg,q2_deg,q3_deg\n")
    for q_deg in q_list:
        f.write("{:.4f},{:.4f},{:.4f}\n".format(*q_deg))
print(f"✅ Đã lưu góc khớp vào '{angle_file}'")




#===== Plot đường đi đầu cuối =====
if q_list:
    positions = [robot.fkine(np.radians(q)).t for q in q_list]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    for i, pos in enumerate(positions):
        print(f"Điểm {i}: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker='o', label='Quỹ đạo đầu cuối')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Đường đi thực tế của đầu cuối robot")
    ax.legend()
    plt.show()
else:
    print("⚠️ Không có điểm nào để vẽ quỹ đạo.")


from scipy.optimize import minimize

# ===== Hàm tối ưu hóa nhiều khớp (vd: q0, q1), giữ nguyên q2 =====
def optimize_joint_subset(q_fixed, joint_indices, T_target):
    def cost(q_vars):
        q_try = q_fixed.copy()
        for i, idx in enumerate(joint_indices):
            q_try[idx] = q_vars[i]
        pos = robot.fkine(q_try).t
        return np.linalg.norm(pos - T_target.t)

    bounds = [robot.links[i].qlim for i in joint_indices]
    x0 = [q_fixed[i] for i in joint_indices]

    res = minimize(cost, x0, bounds=bounds, method='L-BFGS-B')

    if res.success:
        q_opt = q_fixed.copy()
        for i, idx in enumerate(joint_indices):
            q_opt[idx] = res.x[i]
        return q_opt
    else:
        return q_fixed  # fallback

# ===== Tối ưu q0, q1, giữ nguyên q2 cho từng bước =====
q_rad_list = [np.radians(q) for q in q_list]
optimized_q_rad_list = [q_rad_list[0]]  # bắt đầu bằng q đầu tiên

for i in range(1, len(q_rad_list)):
    q_fixed = q_rad_list[i]
    T_target = robot.fkine(q_fixed)
    q_optimized = optimize_joint_subset(q_fixed, joint_indices=[0, 1], T_target=T_target)
    optimized_q_rad_list.append(q_optimized)

# ===== Ghi file kết quả tối ưu hóa =====
optimized_q_deg = [np.degrees(q) for q in optimized_q_rad_list]
angle_file_opt = "square_joint_angles_optimized.txt"
with open(angle_file_opt, "w") as f:
    f.write("q1_deg,q2_deg,q3_deg\n")
    for q in optimized_q_deg:
        f.write("{:.4f},{:.4f},{:.4f}\n".format(*q))
print(f"✅ Đã lưu góc khớp tối ưu vào '{angle_file_opt}'")
