# FaceMeshX
FaceMeshX is a real-time facial expression and head pose detection system using MediaPipe and OpenCV.


# FaceMeshX - Advanced Face & Expression Detection using MediaPipe & OpenCV

This project uses **MediaPipe Face Mesh** and **OpenCV** to detect and visualize advanced facial features and expressions in real-time via webcam. It detects:

- Head direction (left, right, center)
- Head tilt (left, right, straight)
- Eye status (open or closed)
- Smile detection (smiling or not smiling)
- Frames per second (FPS) counter
- Screenshot capture functionality

## Features

- Real-time face mesh overlay on the video feed
- Robust eye aspect ratio (EAR) calculation for eye open/closed detection
- Smile detection based on mouth width-to-height ratio
- Head pose estimation simplified via nose and eye landmarks
- User instructions and screenshot saving with a key press
- Easy to modify thresholds for sensitivity tuning

## Requirements

- Python 3.7+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)

## How to Run

1. Clone the repository or copy the code.
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe
