import cv2
import torch
import numpy as np
import os
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
ENCODER_TYPE = 'vits'  # Change to 'vitb', 'vitl', 'vitg' if needed
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize Input Frame
    input_size = 518  # Model's required input size
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
