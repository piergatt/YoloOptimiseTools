import numpy as np
import cv2

resolution = (192, 192)
# for an 8-bit buffer:
arr = np.fromfile('inp1.bin', dtype=np.uint8).reshape(resolution)
print(arr.shape)
print(np.min(arr), np.max(arr))

cv2.imshow("RGB Frame", arr)
cv2.waitKey(0)
cv2.destroyAllWindows()