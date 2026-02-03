import cv2
import numpy as np

# Load ảnh
img = cv2.imread("./b4a54d494e4ec010995f.jpg")

h, w = img.shape[:2]

# Tạo ảnh RGBA (thêm alpha channel)
img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# 4 điểm gốc
src = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
])

# 4 điểm đích (nghiêng sang phải)
dst = np.float32([
    [0, 0],
    [w-80, 40],
    [w-80, h-40],
    [0, h]
])

# Ma trận perspective
M = cv2.getPerspectiveTransform(src, dst)

# Warp với nền trong suốt
out = cv2.warpPerspective(
    img_rgba,
    M,
    (w, h),
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0, 0)   # Alpha = 0 → trong suốt
)

# Lưu PNG (bắt buộc PNG mới giữ transparency)
cv2.imwrite("output_transparent.png", out)

# cv2.imshow("Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
