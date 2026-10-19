
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_SIZE = 128

df = pd.read_csv(DATA_DIR / "labels.csv")

images = []
boxes = []

for josh, row in df.iterrows():
    image_path = DATA_DIR / "images" / row["filename"]
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load {row['filename']}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    box = np.array([np.clip(row["x1"] / width, 0, 1), np.clip(row["y1"] / height, 0, 1), np.clip(row["x2"] / width, 0, 1), np.clip(row["y2"] / height, 0, 1)], dtype=np.float32)

    # Resize and normalize image
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0

    images.append(image)
    boxes.append(box)

X = np.array(images, dtype=np.float32)
y = np.array(boxes, dtype=np.float32)

np.save(DATA_DIR / "X.npy", X)
np.save(DATA_DIR / "y_labels.npy", y)

print("Image data shape:", X.shape)
print("Bounding-box data shape:", y.shape)
print("Pixel range:", X.min(), "to", X.max())
print("Example bounding box:", y[0])
