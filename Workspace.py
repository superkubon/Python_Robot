import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH

# Định nghĩa robot
deg = np.pi / 180
L1, L2, L3, L4 = 0.15, 0.14, 0, 0.19
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0, offset=-np.pi/3, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L3, alpha=np.pi/2, offset=np.pi, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=L4, a=0, alpha=np.pi/2, offset=0, qlim=[-90 * deg, 90 * deg])
], name='3DOF_Robot')
robot.teach(np.zeros(4))  # Khởi tạo robot với góc khớp ban đầu là 0
# Lấy mẫu góc mỗi khớp (giảm độ phân giải để nhanh)
samples_per_joint = 15
q1 = np.linspace(robot.links[0].qlim[0], robot.links[0].qlim[1], samples_per_joint)
q2 = np.linspace(robot.links[1].qlim[0], robot.links[1].qlim[1], samples_per_joint)
q3 = np.linspace(robot.links[2].qlim[0], robot.links[2].qlim[1], samples_per_joint)
q4 = np.linspace(robot.links[3].qlim[0], robot.links[3].qlim[1], samples_per_joint)

positions = []

# Lặp qua tổ hợp góc khớp (có thể dùng random sampling nếu quá lâu)
for a1 in q1:
    for a2 in q2:
        for a3 in q3:
            for a4 in q4:
                q = [a1, a2, a3, a4]
              
                T = robot.fkine(q)
                pos = T.t
                positions.append(pos)

positions = np.array(positions)

# Vẽ scatter 3D vùng workspace
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='blue', s=1, alpha=0.3)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Workspace of 5 DOF Robot')

# Giữ tỉ lệ đều để thấy đúng hình dạng workspace
max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                      positions[:,1].max()-positions[:,1].min(),
                      positions[:,2].max()-positions[:,2].min()]).max() / 2.0

mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
