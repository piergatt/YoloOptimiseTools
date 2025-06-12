import numpy as np
import cv2

resolution = (192, 192)
# for an 8-bit buffer:
arr = np.fromfile('testInp7.bin', dtype=np.uint8).reshape(resolution)
print(arr.shape)
print(np.unique(arr))

# color_map = np.array([
#     [52,  37,  30],   # Background (B=52, G=37,  R=30)
#     [255, 223, 216],  # ON event   (B=236, G=223, R=216)
#     [201, 126,  64]   # OFF event  (B=201, G=126, R=64)
# ], dtype=np.uint8)

color_map = np.array([
    [0,  0,  0],   # Background (B=52, G=37,  R=30)
    [255, 255, 255],  # ON event   (B=236, G=223, R=216)
    [255, 255,  255]   # OFF event  (B=201, G=126, R=64)
], dtype=np.uint8)

mapped_frame = arr#color_map[arr]
cv2.imwrite("testInp.png", mapped_frame)
cv2.imshow("RGB Frame", mapped_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()