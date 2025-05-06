# %%
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from scipy.special import expit

# %%
# — load class names
cfg = yaml.safe_load(open('models\YOLO_x_nano\st_yolo_x_nano_192_0.33_0.25_config.yaml'))
class_names = cfg['dataset']['class_names']

# - load model
model = tf.keras.models.load_model('models\YOLO_x_nano\st_yolo_x_nano_192_0.33_0.25.h5')
print("expects:", model.input_shape, model.input.dtype)

# - view all model layers
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - output shape: {layer.output_shape}")

# %%
# Load the image
img = Image.open('testImages/YOLO_x_nano/man2.png').convert('RGB')
img = img.resize((192, 192))  # Ensure correct size

# Convert to numpy array and scale to [0, 1]
img_array = np.asarray(img, dtype=np.float32) / 255.0

# If model expects batch dimension, add it:
img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 192, 192, 3)

print(img_array.shape, img_array.dtype, img_array.min(), img_array.max())
plt.imshow(img_array[0])
plt.axis('off')
plt.show()

# %%
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_array)

# %%
p3, p4, p5 = model.predict(img_array)
p3, p4, p5 = p3[0], p4[0], p5[0]
all_boxes = []
all_boxes_flatten = []

for i, p in enumerate([p3, p4, p5]):
    stride = img_array.shape[1]/p.shape[0]
    
    p[..., [0,1,4,5]] = expit(p[..., [0,1,4,5]])
    H, W = p.shape[:2]
    
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    # decode centers
    p[..., 0] = (p[..., 0] + gx) * stride
    p[..., 1] = (p[..., 1] + gy) * stride
    # decode size
    p[..., 2:4] = np.exp(p[..., 2:4]) * stride
    
    all_boxes.append(p)
    all_boxes_flatten.append(p.reshape(-1, p.shape[-1]))
    print(all_boxes_flatten[i].shape)

all_boxes_flatten = np.concatenate(all_boxes_flatten, axis=0)

# %%
# Calculate the maximum and minimum for each of the 6 elements
max_values = np.max(all_boxes[1], axis=(0, 1))
min_values = np.min(all_boxes[1], axis=(0, 1))

# Print the results
for i in range(6):
    print(f"Element {i+1}: Max = {max_values[i]}, Min = {min_values[i]}")

for i in range(all_boxes[1].shape[0]):  # Iterate over rows
    for j in range(all_boxes[1].shape[1]):  # Iterate over columns
        grid_cell_data = all_boxes[1][i, j]  # Shape: (6,)
        df = pd.DataFrame([grid_cell_data], columns=[f"Element {k}" for k in range(1, 7)])
        print(f"Grid Cell ({i}, {j}):")
        print(df)
        print("-" * 50)

# %%
scores = all_boxes_flatten[:, 4] * all_boxes_flatten[:, 5] 
top5_idxs = np.argsort(scores)[-5:][::-1]

# 2. Extract those boxes and their scores
top5_boxes  = all_boxes_flatten[top5_idxs]   # shape (5,6)
top5_scores = scores[top5_idxs]      # shape (5,)

# 3. (Optional) print them out
for rank, (box, score) in enumerate(zip(top5_boxes, top5_scores), start=1):
    x_center, y_center, w, h, obj, cls = box
    print(f"{rank}: score={score:.3f}, center=({x_center:.1f},{y_center:.1f}), "
          f"size=({w:.1f}*{h:.1f})")

# %%
best_idx = np.argmax(scores)
best_box = all_boxes_flatten[best_idx] 

# --- Convert to corner coords ---
cx, cy, w, h = best_box[:4]
x1, y1 = cx - w/2, cy - h/2
x2, y2 = cx + w/2, cy + h/2

# --- Draw the box ---
img_with_box = img.copy()
draw = ImageDraw.Draw(img_with_box)
draw.rectangle([(int(x1), int(y1)), (int(x2), int(y2))],
               outline="white", width=2)
plt.imshow(img_with_box)
plt.axis('off')
plt.show()


