import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH

# --------------------- Robot Definition ---------------------
L = [
    RevoluteDH(d=0.1687, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0.1556, alpha=0),
    RevoluteDH(d=0.2271, a=0, alpha=-np.pi/2),
    RevoluteDH(d=0.16, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0.16, alpha=np.pi/2)
]
robot = DHRobot(L, name='My5DOFRobot')

# --------------------- Geometric IK ---------------------
def geometric_ik(target: SE3):
    d1, a2, d3, d4, a5 = 0.1687, 0.1556, 0.2271, 0.16, 0.16
    px, py, pz = target.t
    R = target.R

    # --- Theta 1 ---
    theta1 = np.arctan2(py, px)

    # --- Wrist position ---
    r = np.hypot(px, py)
    x1 = r - R[0, 2] * a5
    z1 = pz - d1 - R[2, 2] * a5

    # --- Theta 3 ---
    D = (x1**2 + z1**2 - a2**2 - d3**2) / (2 * a2 * d3)
    if abs(D) > 1:
        return None  # unreachable

    theta3 = np.arccos(D)

    # --- Theta 2 ---
    phi1 = np.arctan2(z1, x1)
    phi2 = np.arctan2(d3 * np.sin(theta3), a2 + d3 * np.cos(theta3))
    theta2 = phi1 - phi2

    # --- FK to joint 3 ---
    q123 = [theta1, theta2, theta3]
    T3 = robot.fkine(q123 + [0, 0], end=3)  # Sửa lỗi reshape


    R3 = T3.R
    R35 = R3.T @ R

    # --- Theta 4 and 5 ---
    theta5 = np.arctan2(-R35[2, 0], R35[2, 2])

    # Theta4 (roll around X)
    theta4 = np.arctan2(R35[1, 2], R35[0, 2])

    return np.array([theta1, theta2, theta3, theta4, theta5])

# --------------------- Target and Visualization ---------------------
def make_target(x, y, z):
    z_axis = np.array([0, 0, 1], dtype=float)
    x_temp = np.array([1, 0, 0], dtype=float)
    y_axis = np.cross(z_axis, x_temp)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))
    return SE3.Rt(R, [x, y, z])


def check_ik(target: SE3):
    q = geometric_ik(target)
    if q is None:
        print("❌ Không reach được target này.")
        return

    print("Góc khớp (deg):", np.degrees(q))
    T_check = robot.fkine(q)
    pos_error = np.linalg.norm(target.t - T_check.t)
    print(f"Sai số vị trí: {pos_error:.6f} m")

    # So sánh hướng Z của khớp 4 và end-effector
    T4 = robot.fkine(q, end=3)
    z4 = T4.R[:,2]
    z5 = T_check.R[:,2]
    angle = np.degrees(np.arccos(np.clip(np.dot(z4, z5), -1, 1)))
    print(f"Lệch Z4-Z5: {angle:.2f}°")

    robot.plot(q, block=True)
    plt.title("Geometric IK Result")
    plt.show()

# --------------------- Run Test ---------------------
if __name__ == "__main__":
    T = make_target(0.2, 0.2, 0.2)
    check_ik(T)
    
    T2 = make_target(0.3, 0.3, 0.2)
    check_ik(T2)
    
    T3 = make_target(0.6, 0.0, 0.25)
    check_ik(T3)  # có thể fail nếu nằm ngoài workspace