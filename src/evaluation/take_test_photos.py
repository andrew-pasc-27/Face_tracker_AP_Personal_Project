import pandas as pd
import cv2
import numpy as np
from pathlib import Path

cap = cv2.VideoCapture(0)

IMAGE_DIR = Path("dataset")/"test_images"

records =[]
counter = 0

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:
    return_val, frame = cap.read()

    if not return_val:
        print("Could not read webcam frame.")
        break

    h, w = frame.shape[:2]

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        cv2.destroyAllWindows()
        filename = f"test_img_{counter:04d}.jpg"
        image_path = IMAGE_DIR / filename
        roi = cv2.selectROI("Select ROI", frame, False)
        coords = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
        cv2.imwrite(str(image_path), frame)
        records.append((filename, *map(int, coords)))
        counter += 1
        print(f"Saved {filename}")
        cv2.destroyAllWindows()
    cv2.imshow("Face Center Tracking", frame)

df = pd.DataFrame(records, columns=["filename", "x_min", "y_min", "x_max", "y_max"])
df.to_csv(Path("dataset") / "test_labels.csv", index=False)
cap.release()
cv2.destroyAllWindows()
