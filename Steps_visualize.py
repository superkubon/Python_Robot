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
    RevoluteDH(d=0, a=L4, alpha=np.pi/2, offset=0, qlim=[-120 * deg, 120 * deg]),
], name='3DOF_Robot')
q = np.zeros(3)  # Khởi tạo nghiệm ban đầu là 0
robot.teach(q)

# ===== Hệ số chuyển đổi góc → bước (mm) =====
STEP_CONVERT = {
    'X': 0.355555556,
    'Y': 0.7166666666666667,
    'Z': 0.2222222222222222,

}

# ==== Hàm chuyển step → rad ====
def step_to_rad(x_step, y_step, z_step, a_step):
    q1_deg = -x_step / STEP_CONVERT['X']
    q2_deg = -y_step / STEP_CONVERT['Y']
    q3_deg = z_step / STEP_CONVERT['Z']
    return np.radians([q1_deg, q2_deg, q3_deg])

# ==== Nhập số bước và chuyển thành q ====
q_list = []
num_points = int(input("🔢 Nhập số điểm bạn muốn nhập (theo đơn vị step): "))

for i in range(num_points):
    print(f"\n👉 Nhập thông tin điểm thứ {i+1}:")
    x_step = float(input("  Số bước trục X: "))
    y_step = float(input("  Số bước trục Y: "))
    z_step = float(input("  Số bước trục Z: "))
    
    q_rad = step_to_rad(x_step, y_step, z_step)
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


