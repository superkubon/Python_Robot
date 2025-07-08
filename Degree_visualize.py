import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt

# ==== Định nghĩa robot ====
L1, L2, L3, L4, L5 = 0.211, 0.15, 0.0, 0.23, 0.07
laser_extension = 0.075  # 7.5 cm
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0, offset=-np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=L3, a=0, alpha=np.pi/2, offset=np.pi/2, qlim=[-120 * deg, 120 * deg]),
    RevoluteDH(d=L4, a=0, alpha = np.pi/2, offset=np.pi/2, qlim=[ -180* deg, 180 * deg]),
    RevoluteDH(d=0, a=L4, alpha=np.pi/2, offset=np.pi/2, qlim=[-120 * deg, 120 * deg]),
], name='5DOF_Robot')
q = np.zeros(5)
robot.teach(q)

# ==== STEP → RAD & DEGREE → RAD ====
STEP_CONVERT = {'X': 0.355555556, 'Y': 0.7166666666666667, 'Z': 0.2222222222222222}
def step_to_rad(x_step, y_step, z_step, a_step):
    q1_deg = -x_step / STEP_CONVERT['X']
    q2_deg = -y_step / STEP_CONVERT['Y']
    q3_deg = z_step / STEP_CONVERT['Z']
    return np.radians([q1_deg, q2_deg, q3_deg])

def degree_to_rad(q1_deg, q2_deg, q3_deg, q4_deg=0, q5_deg=0):
    return np.radians([q1_deg, q2_deg, q3_deg, q4_deg, q5_deg])

# ==== Nhập điểm ====
q_list = []
num_points = int(input("🔢 Nhập số điểm bạn muốn nhập (theo đơn vị độ): "))

for i in range(num_points):
    print(f"\n👉 Nhập thông tin góc điểm thứ {i+1} (độ):")
    q1_deg = float(input("  Góc khớp 1: "))
    q2_deg = float(input("  Góc khớp 2: "))
    q3_deg = float(input("  Góc khớp 3: "))
    q_rad = degree_to_rad(q1_deg, q2_deg, q3_deg)
    q_list.append(q_rad)

# ==== Verify động học nghịch ====
# ==== Verify động học nghịch ====
for i, q in enumerate(q_list):
    T = robot.fkine(q)  # FK lấy T (pose)
    robot.teach(q)  # Hiển thị robot với góc q
    sol = robot.ikine_LM(T,q0=q)  # IK để lấy lại q

    print(f"\n📌 Điểm {i+1}:")
    print(" Số góc truyền vào (deg):", np.round(np.degrees(q), 4))
    if sol.success:
        print("  ✅ IK thành công.")
        print("  🔁 Góc tìm được khi tính lại từ động học nghịch (deg):", np.round(np.degrees(sol.q), 4))
        print("  ❗ Sai số từng khớp (deg):", np.round(np.abs(np.degrees(q - sol.q)), 6))
    else:
        print("  ❌ IK thất bại!")


