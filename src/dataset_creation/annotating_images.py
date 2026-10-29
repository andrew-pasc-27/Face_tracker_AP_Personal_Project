import pandas as pd
import cv2
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_DIR = DATA_DIR / "images"

df = pd.read_csv(DATA_DIR / "labels.csv")

print(df.head())

for index, row in df.iterrows():
    filename = row["filename"]
    img = cv2.imread(str(IMAGE_DIR / filename))

    if img is None:
        print(f"Could not load {filename}")
        continue

    print(f"Annotating {index + 1}/{len(df)}: {filename}")

    #use selectroi to create bounding boxes around my face
    roi = cv2.selectROI(f"Face Annotation - {index + 1}", img, False)
    cv2.destroyAllWindows()

    x1, y1, w, h = roi

    # if no box was selected, skip this image
    if w == 0 or h == 0:
        print("Skipped image")
        continue

    x2 = x1 + w
    y2 = y1 + h

    df.loc[index, ["x1", "y1", "x2", "y2"]] = [
        x1, y1, x2, y2
    ]

    # save after EVERY image so progress isn't lost
    df.to_csv(DATA_DIR / "labels.csv", index=False)

print("Annotation session complete!")