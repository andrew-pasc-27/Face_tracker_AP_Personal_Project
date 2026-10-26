
import cv2
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_DIR = DATA_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR/"points.csv")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("could not open webcam")

records = []
image_count = len(df)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("failed to capture frame")
            break

        cv2.imshow("dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        print(frame.shape[:2])
        if key == ord(" "):
            filename = f"img_{image_count:04d}.jpg"
            image_path = IMAGE_DIR / filename

            cv2.imwrite(str(image_path), frame)

            records.append({"filename": filename, "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0})

            image_count += 1
            print(f"Saved {filename}")

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

df = pd.DataFrame(records)

df.to_csv("dataset/labels.csv", index=False)

print(df.head())
print(f"Collected {len(df)} images")
