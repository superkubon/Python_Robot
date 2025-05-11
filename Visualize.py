import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

# ===== Định nghĩa robot 3-DOF =====
L1, L2, L3 = 0.17, 0.17, 0.15
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=np.pi/2),
    RevoluteDH(d=0,  a=L2, alpha=0),
    RevoluteDH(d=0,  a=L3, alpha=0)
], name='3DOF_Robot')

# ===== Danh sách vị trí mục tiêu =====
target_positions = [
    [0.08,  0.04,  0.2],   # trực diện phía trước
    [0.1,  0.1,   0.2],    # hơi lệch phải
    [0.0, -0.15,  0.25],   # phía trái
    [-0.1,  0.1,  0.18],   # sau trái
    [0.15, 0.15,  0.17]    # góc chéo
]

# ===== Chạy kiểm thử với IK =====
for idx, pos in enumerate(target_positions):
    print(f"\n===== Test case {idx + 1} =====")
    print("🎯 Vị trí mục tiêu:", pos)

    T_goal = SE3(*pos)
    ik_result = robot.ikine_LM(T_goal, mask=[1, 1, 1, 0, 0, 0])  # chỉ giải IK vị trí

    if ik_result.success:
        q_ik_rad = ik_result.q
        q_ik_deg = np.degrees(q_ik_rad)
        T_check = robot.fkine(q_ik_rad)
        pos_error = np.linalg.norm(T_check.t - T_goal.t)

        print("✅ IK thành công")
        print("↪️ Góc khớp tìm được (deg):", np.round(q_ik_deg, 2))
        print("📏 Sai số vị trí:", round(pos_error * 1000, 3), "mm")
        print("🔁 FK kiểm tra lại:", np.round(T_check.t, 4))
    else:
        print("❌ IK thất bại – điểm nằm ngoài workspace?")
