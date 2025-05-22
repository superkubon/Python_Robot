import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

# Định nghĩa robot
L1, L2, L3 = 0.2, 0.15, 0.18
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0, offset=-56.33802816901408 * np.pi / 180, qlim=[-75 * deg, 75 * deg]),
    RevoluteDH(d=0, a=L3, alpha=0, offset=92.33610341643583 * np.pi / 180, qlim=[-120 * deg, 120 * deg])
], name='3DOF_Robot')

# Danh sách các bộ giá trị q (đơn vị: độ → rad)
q_deg_list = np.array([
    [-6.3396,-25.0217,-4.0754],
    [-6.3396,-9.6816,12.4923],
    [-31.8486,0.9678,-3.5279],
    [-31.8486,-11.7899,-20.2151],
    [-6.3392,-25.0218,-4.0739]
])

q_rad_list = q_deg_list * deg

# Tính các điểm đầu cuối của robot
points = []
for q in q_rad_list:
    T = robot.fkine(q)
    points.append(T.t)

points = np.array(points)

# Tạo animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0, 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
line, = ax.plot([], [], [], 'o-', lw=2, color='blue')
trail, = ax.plot([], [], [], '--', color='red')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    trail.set_data([], [])
    trail.set_3d_properties([])
    return line, trail

def update(frame):
    q = q_rad_list[frame]
    links = robot.fkine_all(q)
    xyz = np.array([link.t for link in links])
    line.set_data(xyz[:, 0], xyz[:, 1])
    line.set_3d_properties(xyz[:, 2])
    trail.set_data(points[:frame+1, 0], points[:frame+1, 1])
    trail.set_3d_properties(points[:frame+1, 2])
    return line, trail

ani = FuncAnimation(fig, update, frames=len(q_rad_list),
                    init_func=init, blit=False, repeat=True)
plt.show()
