import os
import time

import cv2
import numpy as np
import tensorflow as tf
from custom_giou_loss import GIoULoss

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
CONFIDENCE_THRESHOLD = 0.95
MINIMUM_TARGET_DISTANCE = 5
PAN_GAIN = 1.0
TILT_GAIN = 1.0

model = tf.keras.models.load_model("confidence_face_turret_model.keras")

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

last_sent_target = None
last_send_time = 0.0
smoothed_center = np.array([0.5, 0.5], dtype=np.float32)

while True:
    ret, frame = cap.read()

    if not ret:
        print("unable to read webcam frame")
        break

    height, width = frame.shape[:2]

    #preprocess the images
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    #get model prediction and confidence
    prediction = model.predict(img, verbose=0)
    coordinates = prediction["bbox"][0] 
    confidence = float(prediction["confidence"][0][0])

    x1 = int(np.clip(coordinates[0], 0, 1) * width)
    y1 = int(np.clip(coordinates[1], 0, 1) * height)
    x2 = int(np.clip(coordinates[2], 0, 1) * width)
    y2 = int(np.clip(coordinates[3], 0, 1) * height)

    #only add a quarter of the offset to ensure smoothness
    face_center = np.array([((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height], dtype=np.float32)
    smoothed_center += SMOOTHING * (face_center - smoothed_center)

    face_center_pixels = smoothed_center * np.array([width, height], dtype=np.float32)

    pixel_offset = face_center_pixels - np.array([width / 2, height / 2], dtype=np.float32)

    FOCAL_LENGTH_PX = 1538.6
    angular_offset = np.arctan(pixel_offset / FOCAL_LENGTH_PX)
    #use tangent to calculate the angle of offset for the pan and tilt axes
    angular_offset_deg = np.degrees(angular_offset)

    #converted to degrees and capped
    pan = int(round(SERVO_CENTER - angular_offset_deg[0] * PAN_GAIN))
    tilt = int(round(SERVO_CENTER - angular_offset_deg[1] * TILT_GAIN))

    pan = int(np.clip(pan, SERVO_MIN, SERVO_MAX))
    tilt = int(np.clip(tilt, SERVO_MIN, SERVO_MAX))
    now = time.monotonic()
    if last_sent_target is None:
        target_distance = float("inf")
    else:
        x_distance = face_center_pixels[0] - last_sent_target[0]
        y_distance = face_center_pixels[1] - last_sent_target[1]
        target_distance = np.sqrt(x_distance**2 + y_distance**2)

    #only sent if the model is confident and theres a genuine change in pixels
    if (confidence >= CONFIDENCE_THRESHOLD
        and target_distance >= MINIMUM_TARGET_DISTANCE
        and now - last_send_time >= SEND_INTERVAL):
        servo_serial.write(f"{pan},{tilt}\n".encode("ascii"))
        last_sent_target = face_center_pixels
        last_send_time = now

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"Face: ({x1}, {y1}), ({x2}, {y2})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Servos: pan={pan} tilt={tilt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Center Tracking Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
servo_serial.write(f"{SERVO_CENTER},{SERVO_CENTER}\n".encode("ascii"))
servo_serial.close()
cv2.destroyAllWindows()
