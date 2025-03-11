import cv2
import torch
import numpy as np
import time
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

# Set Device (GPU if available)
DEVICE = 'cpu'

# Depth Model Configuration
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Load Model
ENCODER_TYPE = 'vits'
depth_anything = DepthAnythingV2(**model_configs[ENCODER_TYPE])
depth_anything.load_state_dict(torch.load(
    'D:/PES/lift_auff_narmal/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth',
    map_location=DEVICE
))
depth_anything = depth_anything.to(DEVICE).eval()

# Set Webcam Input
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Colormap for Depth Output
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

# Grid parameters
rows, cols = 2, 3
matrix_result = np.zeros((rows, cols), dtype=int)

# Time tracking for printing matrix every second
last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize Input Frame
    input_size = 518
    frame_resized = cv2.resize(frame, (input_size, input_size))

    # Run Depth Estimation
    with torch.inference_mode():
        depth = depth_anything.infer_image(frame_resized, input_size)

    # Normalize Depth Map
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    # Apply Colormap
    depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # Resize Depth to Match Original Frame
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    # Get Frame Dimensions
    h, w, _ = frame.shape
    cell_h, cell_w = h // rows, w // cols  # Compute cell size

    # Draw Grid on Webcam
    for i in range(1, rows):  # Horizontal lines
        y = i * cell_h
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 2)

    for j in range(1, cols):  # Vertical lines
        x = j * cell_w
        cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 2)

    # Draw Grid on Depth Map
    for i in range(1, rows):  # Horizontal lines
        y = i * cell_h
        cv2.line(depth_colored, (0, y), (w, y), (255, 255, 255), 1)

    for j in range(1, cols):  # Vertical lines
        x = j * cell_w
        cv2.line(depth_colored, (x, 0), (x, h), (255, 255, 255), 1)

    # Color thresholds (in BGR)
    color_ranges = {
        5: ([0, 0, 150], [100, 100, 255]),   # Red
        4: ([0, 100, 200], [100, 200, 255]),  # Orange
        3: ([0, 200, 200], [100, 255, 255]),  # Yellow
        2: ([0, 150, 0], [100, 255, 100]),    # Green
        1: ([100, 0, 0], [255, 100, 100])     # Blue
    }

    # Process all grid cells
    for r in range(rows):
        for c in range(cols):
            x_start, y_start = c * cell_w, r * cell_h
            x_end, y_end = x_start + cell_w, y_start + cell_h
            cell_roi = depth_colored[y_start:y_end, x_start:x_end]

            # Check color in the cell
            detected_value = 0
            for value, (lower, upper) in color_ranges.items():
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(cell_roi, lower, upper)
                if np.any(mask > 0):
                    detected_value = value
                    break

            matrix_result[r, c] = detected_value  # Store detected value

    # Determine Movement Direction
    if matrix_result[0, 1] == 1 and matrix_result[1, 1] == 1:  
        direction = "STRAIGHT"
    else:
        left_side = matrix_result[:, :cols//2].flatten()
        right_side = matrix_result[:, cols//2:].flatten()
        avg_left = np.mean(left_side)
        avg_right = np.mean(right_side)
        direction = "LEFT" if avg_left < avg_right else "RIGHT"

    # Print full matrix every second
    if time.time() - last_print_time >= 1:
        print("\nCurrent Matrix Status:")
        print(matrix_result)
        print(f"Move towards: {direction}")
        last_print_time = time.time()

    # Create Split View
    split_region = np.ones((frame.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([frame, split_region, depth_colored])

    # Display Output
    cv2.imshow("Webcam | Depth Estimation", combined_result)

    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
