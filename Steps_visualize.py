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
# ==== Tỉ lệ chuyển đổi step → độ ====
STEP_CONVERT = {
    'X': 0.355555556,
    'Y': 0.355555556,
    'Z': 0.216666667,
    'A': 0.133333,
}

# ==== Hàm chuyển step → rad ====
def step_to_rad(x_step, y_step, z_step, a_step):
    q1_deg = -x_step / STEP_CONVERT['X']
    q2_deg = -y_step / STEP_CONVERT['Y']
    q3_deg = z_step / STEP_CONVERT['Z']
    q4_deg = a_step / STEP_CONVERT['A']
    return np.radians([q1_deg, q2_deg, q3_deg, q4_deg])

# ==== Nhập số bước và chuyển thành q ====
q_list = []
num_points = int(input("🔢 Nhập số điểm bạn muốn nhập (theo đơn vị step): "))

for i in range(num_points):
    print(f"\n👉 Nhập thông tin điểm thứ {i+1}:")
    x_step = float(input("  Số bước trục X: "))
    y_step = float(input("  Số bước trục Y: "))
    z_step = float(input("  Số bước trục Z: "))
    a_step = float(input("  Số bước trục A: "))
    
    q_rad = step_to_rad(x_step, y_step, z_step, a_step)
    q_list.append(q_rad)

# ==== Hiển thị robot với teach (mô phỏng tuần tự) ====
# ==== Hiển thị robot ====
if len(q_list) == 1:
    robot.teach(q_list[0])
else:
    # Dùng teach từng điểm
    for q in q_list:
        robot.teach(q)

    # Hoặc dùng plot để mô phỏng quỹ đạo
    # robot.plot(q_list, backend='pyplot', block=True)


