import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load uploaded color images
expected_path = "fake.png"  # Green G-code
real_path = "real.png"       # Red robot line

img_expected = cv2.imread(expected_path)
img_real = cv2.imread(real_path)

# Resize to match
img_real_resized = cv2.resize(img_real, (img_expected.shape[1], img_expected.shape[0]))

# === Color-based mask instead of grayscale ===
# Detect green lines in G-code render (expected)
green_lower = np.array([0, 100, 0])
green_upper = np.array([100, 255, 100])
binary_expected = cv2.inRange(img_expected, green_lower, green_upper)

# Detect red lines in real drawing
red_lower1 = np.array([0, 0, 100])
red_upper1 = np.array([100, 100, 255])
binary_real = cv2.inRange(img_real_resized, red_lower1, red_upper1)

# Difference image
diff_image = cv2.absdiff(binary_expected, binary_real)

# Stats
intersection = (cv2.bitwise_and(binary_expected, binary_real) > 0).sum()
union = (cv2.bitwise_or(binary_expected, binary_real) > 0).sum()
expected_sum = (binary_expected > 0).sum()
real_sum = (binary_real > 0).sum()

iou = intersection / union if union > 0 else 0.0
dice = 2 * intersection / (expected_sum + real_sum) if (expected_sum + real_sum) > 0 else 0.0

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].imshow(binary_expected, cmap='gray')
axs[0].set_title("Expected (Green Path Mask)")
axs[1].imshow(binary_real, cmap='gray')
axs[1].set_title("Real (Red Path Mask)")
axs[2].imshow(diff_image, cmap='hot')
axs[2].set_title("Difference")
for ax in axs:
    ax.axis("off")
plt.tight_layout()

(iou, dice)
