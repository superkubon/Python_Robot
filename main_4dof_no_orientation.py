import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import re

# ===== Định nghĩa robot =====
# L1, L2, L3, L4 = 0.1537, 0.1433, 0.077, 0.1203
L1, L2, L3, L4 = 0.1537, 0.1433, 0.0, 0.1943
laser_extension = 0.075  # 7.5 cm
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=np.pi/2, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0, offset=-np.pi/3, qlim=[-75 * deg, 75 * deg]),
    RevoluteDH(d=L3, a=0, alpha=np.pi/2, offset=np.pi, qlim=[-80 * deg, 80 * deg]),
    RevoluteDH(d=L4, a=0, alpha = np.pi/2, offset=0, qlim=[ -180* deg, 180 * deg])

], name='3DOF_Robot')
q = np.zeros(4)  # Khởi tạo nghiệm ban đầu là 0
robot.teach(q)
# ===== Hệ số chuyển đổi góc → bước (mm) =====
STEP_CONVERT = {
    'X': 0.355555,
    'Y': 0.355555,
    'Z': 0.216666,
    'A': 0.133333,
}

# ===== Hàm tính G-code giữ G0/G1 và ưu tiên nghiệm gần nhất =====

def compute_gcode_line(cmd, x, y, z, q0=None, max_attempts=10):
    # T_goal = SE3(x, y, z) * SE3.OA([0, -1, 0], [0, 0, 1]) 
    T_goal = SE3(x, y, z)
    # Base của laser ở xa điểm chiếu (ví dụ, 0.1m sau điểm)
    # laser_offset = 0.1  # laser dài 10cm base_point cách aim_point
    # aim_point = np.array([x, y, z])
    # base_point = aim_point - laser_offset * np.array([0, 1, 0])  # trục X hướng -Y

    # T_laser_base = make_laser_pose(aim_point, base_point)
    # T_laser = T_laser_base * SE3(laser_extension, 0, 0)
    
    for attempt in range(max_attempts):
        ik_result = robot.ikine_LM(T_goal, q0=q0, mask=[1, 1, 1, 0, 0, 0])
        if not ik_result.success:
            continue

        q_deg = np.degrees(ik_result.q)

        if -90 < q_deg[0] < 90 and -80 < q_deg[1] < 80 and -80 < q_deg[2] < 80 :
            x_step = -q_deg[0] * STEP_CONVERT['X']
            y_step = -q_deg[1] * STEP_CONVERT['Y'] 
            z_step = q_deg[2] * STEP_CONVERT['Z']
            a_step = q_deg[3] * STEP_CONVERT['A']
            gcode_line = f"{cmd} X{x_step:.3f} Y{y_step:.3f} Z{z_step:.3f} A{a_step:.3f} F2700"
            return gcode_line, q_deg, ik_result.q

    return None, None, None

# ===== Đọc và xử lý file G-code =====
input_file = "D:/Work/Thesis/Robot_python/input_gcode/circle_gcode_2.nc"

pattern = re.compile(
    r"^(G0|G1|G92)\s+.*?X([-+]?\d*\.?\d+)\s+Y([-+]?\d*\.?\d+)(?:\s+Z([-+]?\d*\.?\d+))?(?:\s+A([-+]?\d*\.?\d+))?",
    re.IGNORECASE
)


gcode_lines = []
q_list = []
q0 = None  # Khởi tạo nghiệm ban đầu là None
def set_axes_equal(ax):
    import numpy as np

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])
x_offset = None  # Khởi tạo offset X là None
z_offset = None  # Khởi tạo offset X là None
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        match = pattern.search(line)
        if not match:
            if line.startswith(("M3", "M5", "G28")):
                gcode_lines.append(line)
            continue

        cmd = match.group(1).upper()
        x_gcode = float(match.group(2))
        y_gcode = float(match.group(3))
        z_gcode = float(match.group(4)) if match.group(4) is not None else 0.0

        SCALE = 1
        if cmd == "G0" and x_offset is None:
            x_offset = x_gcode
            print(f"🌟 Đặt offset X = {x_offset:.4f} từ dòng G1 đầu tiên")
        # # Ghi nhớ offset X từ điểm đầu tiên
        if x_offset is not None:
            x_adj = (x_gcode - x_offset) * SCALE / 1000.0
        else:
            x_adj = x_gcode * SCALE / 1000.0
        #     z_offset = y_gcode  # Giả sử Y là Z trong G-code
        #     print(f"🌟 Offset ban đầu X: {x_offset:.4f} mm")
        
        # x_adj = ((x_gcode - x_offset) / 1000.0) *SCALE  # chuyển từ mm sang mét, chỉ cho X
        # z_adj = ((y_gcode - z_offset) / 1000.0) *SCALE 
        # z = z_adj                  # Z giữ nguyên
        # x = x_adj
        # x = x_gcode * SCALE / 1000.0  # chuyển từ mm sang mét
        z = y_gcode * SCALE / 1000.0  # chuyển từ mm sang mét
        y = 0.2  # giữ nguyên chiều cao robot
        x = x_adj
        gcode_line, q_deg, q_rad = compute_gcode_line(cmd, x, y, z, q0=q0)
        if q_rad is None:
            print(f"❌ IK thất bại tại điểm ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"✅ {gcode_line}")
            print("🔧 Góc khớp (deg): q1 = {:.2f}, q2 = {:.2f}, q3 = {:.2f}, q4 = {:.2f}".format(*q_deg))
            gcode_lines.append(gcode_line)
            q_list.append(q_deg)
            # robot.teach(q_rad)  # Cập nhật robot với nghiệm mới
            q0 = q_rad  # Cập nhật nghiệm cho bước sau

# ===== Ghi file kết quả G-code =====
output_file = "circle_2.txt"
with open(output_file, "w") as f:
    for line in gcode_lines:
        f.write(line + "\n")
print(f"\n✅ Đã lưu {len(gcode_lines)} dòng vào '{output_file}'")

# ===== Ghi file góc khớp ra file riêng =====
angle_file = "gcode_degree.txt"
with open(angle_file, "w") as f:
    f.write("q1_deg,q2_deg,q3_deg,q4_deg\n")
    for q_deg in q_list:
        f.write("{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*q_deg))
print(f"✅ Đã lưu góc khớp vào '{angle_file}'")

#===== Plot đường đi đầu cuối =====
if q_list:
    positions = [robot.fkine(np.radians(q)).t for q in q_list]
    xs = [p[0] for p in positions]  # X thật
    ys = [p[1] for p in positions]  # Z thật gán cho Y plot
    zs = [p[2] for p in positions]  # Y thật gán cho Z plot

    # for i, pos in enumerate(positions):
    #     print(f"Điểm {i}: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
    # for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        # print(f"Plot point {i}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, zs, ys, marker='o', label='Quỹ đạo đầu cuối')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")  # đổi nhãn ở đây
    ax.set_zlabel("Y (m)")  # đổi nhãn ở đây
    # print("Giá trị ys (trục Z plot):", ys)
    ax.set_title("Đường đi thực tế của đầu cuối robot")
    ax.legend()
    ax.view_init(elev=45, azim=45)
    set_axes_equal(ax)
    plt.show()

else:
    print("⚠️ Không có điểm nào để vẽ quỹ đạo.")