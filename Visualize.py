import numpy as np
import matplotlib.pyplot as plt
import re
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

# ===== Robot định nghĩa =====
L1, L2, L3 = 0.17, 0.17, 0.15
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=L2, alpha=0),
    RevoluteDH(d=0, a=L3, alpha=0)
], name='3DOF_Robot')

# ===== Cấu hình =====
Z_DRAWING_PLANE = 0  # Z cố định
INPUT_FILE  = "D:/Work/Thesis/Robot_python/input_gcode/6_gcode.nc"
OUTPUT_FILE = "D:/Work/Thesis/Robot_python/output_gcode/gcode_with_joint_steps.txt"
MAX_POINTS_TO_PROCESS = 2

# ===== Dữ liệu để visualize =====
x_fk, y_fk = [], []
x_target, y_target = [], []

# ===== Xử lý G-code =====
with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
    for idx, line in enumerate(infile):
        if idx >= MAX_POINTS_TO_PROCESS:
            break

        match = re.search(r'(G0|G1).*X([-+]?[0-9]*\.?[0-9]+)\s*Y([-+]?[0-9]*\.?[0-9]+)', line, re.IGNORECASE)
        if not match:
            continue

        g_cmd = match.group(1).upper()
        x_mm = float(match.group(2))
        y_mm = float(match.group(3))
        x, y, z = x_mm / 1000, y_mm / 1000, Z_DRAWING_PLANE

        T_target = SE3(x, y, z)
        ik_result = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])

        if not ik_result.success:
            print(f"⚠️ IK fail at line {idx + 1}")
            continue

        q_deg = np.degrees(ik_result.q)

        # ===== Chuyển sang bước (step) theo công thức bạn yêu cầu =====
        x_step = q_deg[0] * 0.355555
        y_step = q_deg[1] * 0.355555
        z_step = q_deg[2] * 0.216666
        outfile.write(f"{g_cmd} X{x_step:.3f} Y{y_step:.3f} Z{z_step:.3f} F2700\n")

        # FK → để kiểm tra
        T_fk = robot.fkine(ik_result.q)
        pos = T_fk.t
        x_target.append(x_mm)
        y_target.append(y_mm)
        x_fk.append(pos[0] * 1000)
        y_fk.append(pos[1] * 1000)

# ===== Vẽ đường đi =====
plt.figure(figsize=(8, 8))
plt.plot(x_target, y_target, 'r--o', label="G-code Target Points")
plt.plot(x_fk, y_fk, 'b-o', label="FK after IK")
plt.title("Compare G-code vs Actual Path")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.gca().set_aspect('equal')
plt.grid(True)
plt.legend()
plt.show()
