import pandas as pd
import cv2
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_DIR = DATA_DIR / "images"

df = pd.read_csv(DATA_DIR / "points.csv")

print(df.head())

for index, row in df.iterrows():
    filename = row["filename"]
    x, y = row["center_x"], row["center_y"]
    if x == 0 and y == 0:
        img = cv2.imread(str(IMAGE_DIR / filename))

        if img is None:
            print(f"Could not load {filename}")
            continue

        print(f"Annotating {index + 1}/{len(df)}: {filename}")

        roi = cv2.selectROI(f"Face Annotation - {index + 1}", img, False)
        cv2.destroyAllWindows()

        x1, y1, w, h = roi

        if w == 0 or h == 0:
            print("Skipped image")
            continue

        df.loc[index, ["center_x", "center_y"]] = [x1, y1]
        df.to_csv(DATA_DIR / "points.csv", index=False)
    else:
        print(f"Skipping {index + 1}/{len(df)}: {filename} (already annotated)")

print("Annotation session complete!")