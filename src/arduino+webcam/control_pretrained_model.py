import os
import time

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


SERIAL_PORT = os.getenv("FACE_TURRET_SERIAL", "/dev/cu.usbserial-2110")
SERIAL_BAUD = 115200
SERVO_MIN = 10
SERVO_MAX = 170
SERVO_CENTER = 90
SEND_INTERVAL = 1 / 20
SMOOTHING = 0.25
CONFIDENCE_THRESHOLD = 0.6
MINIMUM_TARGET_DISTANCE = 5
PAN_GAIN = 1.0
TILT_GAIN = 1.0
CONTROL_GAIN = 0.35
DIAGONAL_FOV_DEG = 55.0
MODEL_PATH = "face_detection_yunet_2023mar.onnx"

if not hasattr(cv2, "FaceDetectorYN_create"):
    raise SystemExit(
        "This OpenCV build does not include FaceDetectorYN. Install opencv-contrib-python."
    )

try:
    face_detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        CONFIDENCE_THRESHOLD,
        0.3,
        5000,
    )
except cv2.error as error:
    raise SystemExit(f"unable to load YuNet ONNX model {MODEL_PATH}: {error}")

if serial is None:
    raise SystemExit("pyserial is required for servo control. Install it with: python -m pip install pyserial")

try:
    servo_serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    time.sleep(2)
    print(f"connected to servo controller on {SERIAL_PORT} at {SERIAL_BAUD} baud")
except serial.SerialException as error:
    raise SystemExit(f"unable to open servo serial port {SERIAL_PORT}: {error}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("unable to open webcam")
    servo_serial.close()
    raise SystemExit(1)

last_send_time = 0.0
smoothed_center = np.array([0.5, 0.5], dtype=np.float32)
pan_target = float(SERVO_CENTER)
tilt_target = float(SERVO_CENTER)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("unable to read webcam frame")
            break

        height, width = frame.shape[:2]
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(frame)

        if faces is not None and len(faces) > 0:
            face = max(faces, key=lambda detection: detection[-1])
            x, y, box_width, box_height = face[:4]
            confidence = float(face[-1])

            x1 = int(np.clip(x, 0, width - 1))
            y1 = int(np.clip(y, 0, height - 1))
            x2 = int(np.clip(x + box_width, 0, width - 1))
            y2 = int(np.clip(y + box_height, 0, height - 1))

            face_center = np.array(
                [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height],
                dtype=np.float32,
            )
            smoothed_center += SMOOTHING * (face_center - smoothed_center)
            face_center_pixels = smoothed_center * np.array([width, height], dtype=np.float32)
            pixel_offset = face_center_pixels - np.array(
                [width / 2, height / 2], dtype=np.float32
            )
            diagonal_pixels = np.hypot(width, height)
            focal_length_px = (diagonal_pixels / 2) / np.tan(
                np.radians(DIAGONAL_FOV_DEG / 2)
            )
            angular_offset = np.arctan(pixel_offset / focal_length_px)
            angular_offset_deg = np.degrees(angular_offset)

            now = time.monotonic()
            if (
                confidence >= CONFIDENCE_THRESHOLD
                and np.linalg.norm(pixel_offset) >= MINIMUM_TARGET_DISTANCE
                and now - last_send_time >= SEND_INTERVAL
            ):
                pan_target -= angular_offset_deg[0] * PAN_GAIN * CONTROL_GAIN
                tilt_target -= angular_offset_deg[1] * TILT_GAIN * CONTROL_GAIN
                pan_target = float(np.clip(pan_target, SERVO_MIN, SERVO_MAX))
                tilt_target = float(np.clip(tilt_target, SERVO_MIN, SERVO_MAX))
                pan = int(round(pan_target))
                tilt = int(round(tilt_target))
                servo_serial.write(f"{pan},{tilt}\n".encode("ascii"))
                last_send_time = now
            else:
                pan = int(round(pan_target))
                tilt = int(round(tilt_target))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Face: ({x1}, {y1}), ({x2}, {y2})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Servos: pan={pan} tilt={tilt}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Face Center Tracking Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    servo_serial.write(f"{SERVO_CENTER},{SERVO_CENTER}\n".encode("ascii"))
    servo_serial.close()
    cv2.destroyAllWindows()
