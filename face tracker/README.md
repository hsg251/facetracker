# THANHDAN VISION

**Real-Time Computer Vision & Hand Gesture Recognition System**

THANHDAN VISION is a Python-based computer vision project that combines face detection and hand gesture recognition to create an interactive camera system.

## Features

- Real-time face detection using OpenCV YuNet
- Hand gesture recognition using MediaPipe
- Live hand landmark tracking
- DroidCam WiFi camera support
- Gesture-based keyboard control
- Real-time FPS monitoring
- Chrome Dino game integration

## Gesture Controls

| Gesture | Action |
|---|---|
| Closed Fist ✊ | Jump (Space) |
| Open Palm ✋ | Normal |
| No Hand | No action |

## Built With

- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

## Project Structure

```text
face tracker/
│
├── main.py
├── requirements.txt
├── README.md
│
├── face_detection_yunet_2023mar.onnx
└── gesture_recognizer.task
```

## Installation

**1. Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
cd "face tracker"
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download AI Models**

- YuNet: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- MediaPipe Gesture Recognizer: [Download Model](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)

Place both model files in the project directory.

## Configuration

Connect your phone and computer to the same WiFi network.

Run the program and enter your DroidCam IP address when prompted.

Default port: `4747`

## Run

```bash
python main.py
```

## Requirements

- Python 3.12 recommended
- Windows / Linux
- DroidCam application
- Working webcam or WiFi camera
- Chrome browser for Dino game control

## Developer

**THANHDAN**

Project: face tracker skibidi

Version: 2.0

---

*Built with Python and Computer Vision.*