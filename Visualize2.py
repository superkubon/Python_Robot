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

STEP_CONVERT = {
    'X': 0.355555,
    'Y': 0.355555,
    'Z': 0.216666
}

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
            return gcode_line, q_deg, ik_result.q

    return None, None, None

# ===== Đọc và xử lý file G-code =====
input_file = "D:/Work/Thesis/Robot_python/input_gcode/6_gcode.nc"
pattern = re.compile(r"^(G0|G1)\s+.*?X([-+]?\d*\.?\d+)\s+Y([-+]?\d*\.?\d+)(?:\s+Z([-+]?\d*\.?\d+))?", re.IGNORECASE)

gcode_lines = []
q_list = []
original_coords = []  # ← lưu gốc điểm đầu vào
positive_z_log = []   # ← lưu các dòng sẽ ghi vào file txt

with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue
        cmd = match.group(1).upper()
        x_mm = float(match.group(2))
        y_mm = float(match.group(3))
        z_mm = float(match.group(4)) if match.group(4) is not None else 0.0

        # Lưu tọa độ gốc để ghi ra nếu cần
        original_coords.append((x_mm, y_mm, z_mm))

        # mm → m + offset
        x = -x_mm / 1000.0 - 0.09175
        y = -y_mm / 1000.0 - 0.08498
        z = 0.0

        gcode_line, q_deg, q_rad = compute_gcode_line(cmd, x, y, z)
        if q_rad is None:
            print(f"❌ IK thất bại tại điểm ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"✅ {gcode_line}")
            print("🔧 Góc khớp (deg): q1 = {:.2f}, q2 = {:.2f}, q3 = {:.2f}".format(*q_deg))
            gcode_lines.append(gcode_line)
            q_list.append(q_rad)

            # FK → nếu Z > 0 thì ghi vào log
            fk_pos = robot.fkine(q_rad).t
            if fk_pos[2] > 0:
                log_entry = (
                    f"Input Point (mm): X={x_mm:.2f}, Y={y_mm:.2f}, Z={z_mm:.2f} → "
                    f"FK.Z={fk_pos[2]:.5f} | "
                    f"Joint Angles (deg): q1={q_deg[0]:.2f}, q2={q_deg[1]:.2f}, q3={q_deg[2]:.2f}"
                )
                positive_z_log.append(log_entry)

# ===== Ghi file kết quả G-code =====
output_file = "converted_output_with_g0g1.gcode"
with open(output_file, "w") as f:
    for line in gcode_lines:
        f.write(line + "\n")
print(f"\n✅ Đã lưu {len(gcode_lines)} dòng vào '{output_file}'")

# ===== Ghi file log các điểm có Z > 0 sau FK =====
log_file = "joint_angles_with_fk_z_positive.txt"
with open(log_file, "w", encoding="utf-8") as f:
    for entry in positive_z_log:
        f.write(entry + "\n")
print(f"✅ Đã lưu {len(positive_z_log)} dòng vào '{log_file}' (Z > 0 sau FK)")

# ===== Plot đường đi đầu cuối =====
if q_list:
    positions = [robot.fkine(q).t for q in q_list]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [0 for _ in positions]  # hoặc [p[2] nếu bạn muốn xem Z thật sự]

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
