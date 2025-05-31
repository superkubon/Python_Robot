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

<<<<<<< HEAD
            current_steps = [s for s, _ in steps_with_backlash]
            delta_steps = [abs(c - p) for c, p in zip(current_steps, prev_steps)]
            motion_mm = [ds / 500.0 for ds in delta_steps]
            prev_steps = current_steps

            converted_gcode.append(
                f"G1 X{motion_mm[0]:.2f} Y{motion_mm[1]:.2f} Z{motion_mm[2]:.2f} "
                f"A{motion_mm[3]:.2f} B{motion_mm[4]:.2f} F2700"
            )
            successful += 1

    with open(output_path, "w") as fout:
        fout.write("\n".join(converted_gcode))
    print(f"✅ Generated {successful}/{total_lines} G-code lines. Saved to {output_path}")

# -------------------- Vẽ đường đi đầu vào --------------------
def visualize_input_path(input_path):
    x_vals = []
    y_vals = []

    with open(input_path, "r") as f:
        for line in f:
            result = parse_gcode_line(line)
            if not result:
                continue
            x, y = result
            x_vals.append(x + offset_x)
            y_vals.append(y + offset_y)

    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue')
    plt.title("Offset G-code Drawing Path (in mm)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# -------------------- Kiểm tra IK-FK --------------------
def verify_ik_with_fk(input_path):
    total = 0
    passed = 0
    tolerance = 0.05  # mm

    with open(input_path, "r") as f:
        for line in f:
            result = parse_gcode_line(line)
            if not result:
                continue

            x, y = result
            x += offset_x
            y += offset_y
            z = Z_DRAWING_PLANE
            T_target = make_target_position_only(x, y, z)
            sol = robot.ikine_LM(T_target, mask=[1, 1, 1, 0, 0, 0])
            if not sol.success:
                print(f"⚠️ IK failed at ({x:.3f}, {y:.3f}, {z:.3f})")
                continue

            T_check = robot.fkine(sol.q)
            pos_fk = T_check.t * 1000  # m → mm
            error = np.linalg.norm(pos_fk - np.array([x, y, z]))

            if error < tolerance:
                passed += 1
            else:
                print(f"❌ Line {total+1}: Error = {error:.4f} mm")
                print(f"   Input  = ({x:.2f}, {y:.2f}, {z:.2f})")
                print(f"   FK out = ({pos_fk[0]:.2f}, {pos_fk[1]:.2f}, {pos_fk[2]:.2f})")

            total += 1

    print(f"✅ IK → FK verified: {passed}/{total} points passed within {tolerance:.2f} mm tolerance.")

# -------------------- Chạy toàn bộ --------------------
if __name__ == "__main__":
    input_gcode = "D:/Work/Thesis/Robot_python/input_gcode/circle_gcode_2.nc"
    output_gcode = "D:/Work/Thesis/Robot_python/output_gcode/gcode_motor_movement_relative.txt"
    
    convert_gcode_to_motor_movement(input_gcode, output_gcode)
    visualize_input_path(input_gcode)
    verify_ik_with_fk(input_gcode)
=======
ani = FuncAnimation(fig, update, frames=len(q_rad_list),
                    init_func=init, blit=False, repeat=True)
plt.show()
>>>>>>> 469cc923943243eb509a174999dcfbd3857973c4
