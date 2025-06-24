import numpy as np
import cv2

# Resolution of image captured
resolution = (192, 192)

# for an 8-bit buffer:
arr = np.fromfile('inp1.bin', dtype=np.uint8).reshape(resolution)
print(arr.shape)
print(np.unique(arr))

color_map = np.array([
    [0,  0,  0],   # Background 
    [0, 0, 255],  # ON event
    [255, 0,  0]   # OFF event
], dtype=np.uint8)

# Map the values in arr to the color_map
mapped_frame = arr #color_map[arr]

# Denoise if needed
#denoised_frame = cv2.fastNlMeansDenoisingColored(mapped_frame, None, 10, 10, 7, 21)
#cv2.imshow("Denoised Frame", denoised_frame)

cv2.imwrite("camOutputs/alltogehter.png", mapped_frame)
cv2.imshow("RGB Frame", mapped_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()