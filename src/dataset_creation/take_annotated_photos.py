import pandas as pd

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

model = cv2.FaceDetectorYN_create("/Users/dr/face-turret/face_detection_yunet_2023mar.onnx", config="", input_size=(960, 1280), score_threshold=0.6, nms_threshold=0.3, top_k=5000)

cap = cv2.VideoCapture(0)

IMAGE_DIR = Path("dataset")/"images"

df = pd.read_csv("dataset/points.csv")

counter = len(df)

records =[]

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    h, w = frame.shape[:2]
    target = None
    save_frame = None
    model.setInputSize((w, h))
    john_cena, faces = model.detect(frame)
    
    if faces is not None:
        for face in faces:
            x, y, box_w, box_h = map(int, face[:4])
            center_x = int((x + box_w) / 2)
            center_y = int((y + box_h) / 2)
            target = (center_x, center_y)

            save_frame = frame.copy()
                
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
    cv2.imshow("Face Center Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" ") and target is not None and save_frame is not None:
        filename = f"img_{counter:04d}.jpg"
        image_path = IMAGE_DIR / filename
                
        cv2.imwrite(str(image_path), save_frame)
                
        df.loc[counter, ["filename", "center_x", "center_y"]] = [filename, *map(int, target)]
        df.to_csv(Path("dataset") / "points.csv", index=False)
                
        counter += 1
        print(f"Saved {filename}")


cap.release()
cv2.destroyAllWindows()
