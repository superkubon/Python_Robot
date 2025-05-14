import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import re

# ===== Định nghĩa robot =====
L1, L2, L3 = 0.2, 0.15, 0.18
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=-np.pi, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0,         qlim=[-75 * deg, 75 * deg]),
    RevoluteDH(d=0, a=L3, alpha=0,         qlim=[-120 * deg, 120 * deg])
], name='3DOF_Robot')

# ===== Hệ số chuyển đổi góc → bước (mm) =====
STEP_CONVERT = {
    'X': 0.355555,
    'Y': 0.355555,
    'Z': 0.216666
}

# ===== Hàm tính G-code giữ G0/G1 và lưu nghiệm hợp lệ =====
def compute_gcode_line(cmd, x, y, z, max_attempts=10):
    T_goal = SE3(x, y, z)

    for attempt in range(max_attempts):
        ik_result = robot.ikine_LM(T_goal, mask=[1, 1, 1, 0, 0, 0])
        if not ik_result.success:
            continue

        q_deg = np.degrees(ik_result.q)

        if -90 < q_deg[0] < 90 and -80 < q_deg[1] < 80 and -90 < q_deg[2] < 90:
            x_step = -q_deg[0] * STEP_CONVERT['X']
            y_step = -q_deg[1] * STEP_CONVERT['Y']
            z_step = q_deg[2] * STEP_CONVERT['Z']
            gcode_line = f"{cmd} X{x_step:.3f} Y{y_step:.3f} Z{z_step:.3f}"
            return gcode_line, q_deg, ik_result.q  # ← Trả thêm nghiệm q (rad)

    return None, None, None

# ===== Đọc và xử lý file G-code =====
input_file = "D:/Work/Thesis/Robot_python/input_gcode/6_gcode.nc"
pattern = re.compile(r"^(G0|G1)\s+.*?X([-+]?\d*\.?\d+)\s+Y([-+]?\d*\.?\d+)(?:\s+Z([-+]?\d*\.?\d+))?", re.IGNORECASE)

gcode_lines = []
q_list = []  # ← Danh sách q để plot
with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue
        cmd = match.group(1).upper()
        x = float(match.group(2))
        y = float(match.group(3))
        z = float(match.group(4)) if match.group(4) is not None else 0.0

        # 🔄 mm → m + offset
        x = -x / 1000.0 - 0.09175
        y = -y / 1000.0 - 0.08498
        z =  0 / 1000.0

        gcode_line, q_deg, q_rad = compute_gcode_line(cmd, x, y, z)
        if q_rad is None:
            print(f"❌ IK thất bại tại điểm ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"✅ {gcode_line}")
            print("🔧 Góc khớp (deg): q1 = {:.2f}, q2 = {:.2f}, q3 = {:.2f}".format(*q_deg))
            gcode_lines.append(gcode_line)
            q_list.append(q_rad)

# ===== Ghi file kết quả =====
output_file = "converted_output_with_g0g1.gcode"
with open(output_file, "w") as f:
    for line in gcode_lines:
        f.write(line + "\n")
print(f"\n✅ Đã lưu {len(gcode_lines)} dòng vào '{output_file}'")

# ===== Plot đường đi đầu cuối =====
if q_list:
    positions = [robot.fkine(q).t for q in q_list]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [0 for p in positions]
    
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
