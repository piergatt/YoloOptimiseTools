import numpy as np
import cv2
import os

input_dir = 'val'
output_dir = 'val_rgb'

npy_file = 'val/frame0000000.npy'

resolution = (346, 260)

os.makedirs(output_dir, exist_ok=True)

cv2.namedWindow("RGB Frame", cv2.WINDOW_NORMAL)

for npy_file in os.listdir(input_dir):
    if npy_file.endswith('.npy'):
        # Load the events from the .npy file
        events = np.load(os.path.join(input_dir, npy_file))

        # Initialize an RGB frame (height, width, 3)
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        # Iterate through the events
        for event in events:
            t, x, y, p = event  # Unpack the event fields
            if p == 1:  # Positive polarity
                frame[y, x] = [255, 255, 255]  # White
            elif p == 0:  # Negative polarity
                frame[y, x] = [139, 0, 0]  # Dark blue (BGR format)

        # Save the frame as an image in the output directory
        output_file = os.path.join(output_dir, f"{os.path.splitext(npy_file)[0]}.png")
        cv2.imshow("RGB Frame", frame)
        cv2.imwrite(output_file, frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit early
            break
        
cv2.destroyAllWindows()
print(f"All frames have been converted and saved to '{output_dir}'.")