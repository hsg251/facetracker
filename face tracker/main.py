import cv2
import time
import os
import mediapipe as mp
import pyautogui

# Config
ip = "enter your droidcam wifi ip"
port = 4747

face_model = "face_detection_yunet_2023mar.onnx"
hand_model = "gesture_recognizer.task"

for file in [face_model, hand_model]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Model not found: {file}")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# Connect camera
cap = cv2.VideoCapture(
    f"http://{ip}:{port}/video",
    cv2.CAP_FFMPEG
)

if not cap.isOpened():
    raise RuntimeError("Cannot connect to DroidCam")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Cannot read camera")

height, width = frame.shape[:2]

# Face detection
ai_width = 320
ai_height = 240

face_ai = cv2.FaceDetectorYN.create(
    face_model,
    "",
    (ai_width, ai_height),
    0.65,
    0.3,
    1000
)

# Hand detection
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

options = GestureOptions(
    base_options=BaseOptions(model_asset_path=hand_model),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55
)

hand_ai = GestureRecognizer.create_from_options(options)

# Settings
face_delay = 0.25
hand_delay = 0.12
jump_delay = 0.8
confidence_limit = 0.65

faces = []
hand_points = None

gesture = "NO HAND"
score = 0

last_face = 0
last_hand = 0
last_jump = 0
previous_gesture = ""

last_frame = time.monotonic()
fps = 0

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

print("face tracker AI")
print("Camera: DroidCam")
print("Face: YuNet")
print("Hand: MediaPipe")
print("Control: Thumb Up = Jump")
print("Press Q to exit")

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera disconnected")
            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        now = time.monotonic()

        # Detect face
        if now - last_face >= face_delay:

            small = cv2.resize(
                frame,
                (ai_width, ai_height),
                interpolation=cv2.INTER_AREA
            )

            _, results = face_ai.detect(small)
            faces = []

            if results is not None:
                for face in results:
                    x, y, w, h = face[:4]

                    x = int(x * width / ai_width)
                    y = int(y * height / ai_height)
                    w = int(w * width / ai_width)
                    h = int(h * height / ai_height)

                    faces.append((x, y, w, h))

            last_face = now

        # Detect hand
        if now - last_hand >= hand_delay:

            small = cv2.resize(
                frame,
                (ai_width, ai_height)
            )

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = hand_ai.recognize_for_video(
                image,
                int(now * 1000)
            )

            hand_points = None
            gesture = "NO HAND"
            score = 0

            if result.hand_landmarks:
                hand_points = result.hand_landmarks[0]

                if result.gestures and result.gestures[0]:
                    category = result.gestures[0][0]
                    gesture = category.category_name
                    score = category.score

            last_hand = now

        # Dino control
        is_fist = (
            gesture == "Closed_Fist"
            and score >= confidence_limit
        )

        if (
            is_fist
            and previous_gesture != "Closed_Fist"
            and now - last_jump >= jump_delay
        ):
            pyautogui.press("space")
            last_jump = now
            print("Dino Jump!")

        previous_gesture = "Closed_Fist" if is_fist else ""

        # Draw faces
        for x, y, w, h in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Draw hand
        if hand_points:
            points = []

            for point in hand_points:
                px = int(point.x * width)
                py = int(point.y * height)
                points.append((px, py))

            for a, b in connections:
                cv2.line(
                    frame,
                    points[a],
                    points[b],
                    (255, 180, 0),
                    2
                )

            for px, py in points:
                cv2.circle(
                    frame,
                    (px, py),
                    4,
                    (0, 255, 255),
                    -1
                )

        # FPS
        elapsed = now - last_frame

        if elapsed > 0:
            current_fps = 1 / elapsed
            fps = current_fps if fps == 0 else fps * 0.85 + current_fps * 0.15

        last_frame = now

        # Display
        cv2.rectangle(
            frame,
            (0, 0),
            (340, 185),
            (20, 20, 20),
            -1
        )

        info = [
            ("face tracker AI", (255, 255, 255)),
            (f"FPS: {int(fps)}", (0, 255, 0)),
            (f"Faces: {len(faces)}", (0, 255, 255)),
            (f"Gesture: {gesture}", (255, 200, 0)),
            (f"Confidence: {score:.2f}", (255, 255, 255)),
            ("THUMB UP = JUMP", (0, 255, 0))
        ]

        for i, (text, color) in enumerate(info):
            cv2.putText(
                frame,
                text,
                (10, 25 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        cv2.imshow("face tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hand_ai.close()
    cv2.destroyAllWindows()

print("face tracker stopped")
