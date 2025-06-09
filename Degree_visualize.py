import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt

# ==== Định nghĩa robot ====
L1, L2, L3, L4 = 0.1537, 0.1433, 0.0, 0.1443
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0, offset=-np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=L3, a=0, alpha=np.pi/2, offset=np.pi/2, qlim=[-120 * deg, 120 * deg]),
    RevoluteDH(d=L4, a=0, alpha=np.pi/2, offset=0, qlim=[-180 * deg, 180 * deg])
], name='3DOF_Robot')
robot.teach(np.zeros(4))  # Hiển thị robot với nghiệm ban đầu là 0

# ==== Hàm chuyển degree → rad ====
def degree_to_rad(q1_deg, q2_deg, q3_deg, q4_deg):
    return np.radians([q1_deg, q2_deg, q3_deg, q4_deg])

# ==== Nhập số điểm và chuyển thành q ====
q_list = []
num_points = int(input("🔢 Nhập số điểm bạn muốn nhập (theo đơn vị độ): "))

for i in range(num_points):
    print(f"\n👉 Nhập thông tin góc điểm thứ {i+1} (độ):")
    q1_deg = float(input("  Góc khớp 1: "))
    q2_deg = float(input("  Góc khớp 2: "))
    q3_deg = float(input("  Góc khớp 3: "))
    q4_deg = float(input("  Góc khớp 4: "))
    
    q_rad = degree_to_rad(q1_deg, q2_deg, q3_deg, q4_deg)
    q_list.append(q_rad)

# ==== Hiển thị robot với teach (mô phỏng tuần tự) ====
if len(q_list) == 1:
    robot.teach(q_list[0])
else:
    for q in q_list:
        robot.teach(q)

    # Hoặc dùng plot để mô phỏng quỹ đạo
    # robot.plot(q_list, backend='pyplot', block=True)
