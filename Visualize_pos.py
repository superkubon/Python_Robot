import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

# ===== Định nghĩa robot =====
L1, L2, L3 = 0.2, 0.15, 0.18
deg = np.pi / 180
robot = DHRobot([
    RevoluteDH(d=L1, a=0, alpha=-np.pi/2, offset=0, qlim=[-90 * deg, 90 * deg]),
    RevoluteDH(d=0, a=L2, alpha=0,  offset=-np.pi/3,       qlim=[-100 * deg, 100 * deg]),
    RevoluteDH(d=0, a=L3, alpha=0, offset = np.pi/2,         qlim=[-120 * deg, 120 * deg])
], name='3DOF_Robot')

# ===== Hệ số chuyển đổi góc → bước (mm) =====
STEP_CONVERT = {
    'X': 0.355555,
    'Y': 0.355555,
    'Z': 0.216666
}

# ===== Khởi tạo hoặc nhập tay tư thế ban đầu =====
manual = input("🛠 Bạn có muốn nhập góc tay để xem robot.teach()? (y/n): ").strip().lower()
if manual == 'y':
    try:
        q1_deg = float(input("🔧 Nhập q1 (deg): "))
        q2_deg = float(input("🔧 Nhập q2 (deg): "))
        q3_deg = float(input("🔧 Nhập q3 (deg): "))
        q = np.radians([q1_deg, q2_deg, q3_deg])
    except ValueError:
        print("❌ Góc không hợp lệ. Dùng tư thế mặc định (0,0,0)")
        q = np.zeros(3)
else:
    q = np.zeros(3)

robot.teach(q)

# ===== Hàm tính G-code cho 1 điểm với kiểm tra góc khớp nằm trong giới hạn =====
def compute_gcode_line(x, y, z, max_attempts=10):
    T_goal = SE3(x, y, z)
    
    for _ in range(max_attempts):
        ik_result = robot.ikine_LM(T_goal, mask=[1, 1, 1, 0, 0, 0])
        if not ik_result.success:
            continue
        
        q_deg = np.degrees(ik_result.q)
        
        # Kiểm tra góc có nằm trong giới hạn qlim không
        within_limits = all(
            low <= val <= high 
            for val, (low, high) in zip(q_deg, [( -90, 90), (-90, 90), (-100, 100)])
        )
        if within_limits:
            x_step = -q_deg[0] * STEP_CONVERT['X']
            y_step = q_deg[2] * STEP_CONVERT['Y']
            z_step = q_deg[1] * STEP_CONVERT['Z']
            gcode_line = f"G1 X{x_step:.3f} Y{y_step:.3f} Z{-z_step:.3f}"
            return gcode_line, q_deg, None

    return None, None, f"❌ IK thất bại hoặc góc vượt giới hạn tại điểm ({x:.3f}, {y:.3f}, {z:.3f})"


# ===== Nhập số lượng điểm =====
try:
    n = int(input("🔢 Nhập số lượng điểm cần tính: "))
except ValueError:
    print("❌ Giá trị không hợp lệ.")
    exit()

# ===== Nhập từng điểm và xử lý =====
gcode_lines = []
for i in range(n):
    print(f"\n📍 Nhập tọa độ điểm {i+1}:")
    try:
        x = float(input("  ➤ X (m): "))
        y = float(input("  ➤ Y (m): "))
        z = float(input("  ➤ Z (m): "))
    except ValueError:
        print("❌ Giá trị không hợp lệ. Bỏ qua điểm này.")
        continue

    gcode_line, q_deg, error = compute_gcode_line(x, y, z)
    if error:
        print(error)
    else:
        print("✅", gcode_line)
        print("🔧 Góc khớp (deg): q1 = {:.2f}, q2 = {:.2f}, q3 = {:.2f}".format(*q_deg))
        gcode_lines.append(gcode_line)

# ===== (Tùy chọn) Lưu file =====
save = input("\n💾 Bạn có muốn lưu file G-code không? (y/n): ").strip().lower()
if save == 'y':
    filename = input("📁 Nhập tên file để lưu (VD: output.gcode): ").strip()
    with open(filename, "w") as f:
        for line in gcode_lines:
            f.write(line + "\n")
    print(f"✅ Đã lưu {len(gcode_lines)} dòng vào {filename}")
