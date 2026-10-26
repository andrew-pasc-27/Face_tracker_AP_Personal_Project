
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_SIZE = 128

df = pd.read_csv(DATA_DIR / "blank_labels.csv")

IMAGE_DIR = Path("dataset/blank_images")

records = []

for john_cena_du_du_du_du, row in df.iterrows():
    image_path = IMAGE_DIR / row["filename"]
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"could not load {row['filename']}")
        continue

    h, w = image.shape[:2]
    x1, y1, x2, y2 = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

    flipped = cv2.flip(image, 1)
    new_x1 = w - 1 - x2
    new_x2 = w - 1 - x1


    cv2.imwrite(str(IMAGE_DIR / f"lr_{row['filename']}"), flipped)

    records.append([f"lr_{row['filename']}", new_x1, y1, new_x2, y2])

    flipped = cv2.flip(image, 0)
    new_y1 = h - 1 - y2
    new_y2 = h - 1 - y1

    cv2.imwrite(str(IMAGE_DIR / f"ud_{row['filename']}"), flipped)

    records.append([f"ud_{row['filename']}", x1, new_y1, x2, new_y2])

    bright = cv2.convertScaleAbs(image, alpha=1.0, beta=40)

    dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-40)

    bright_name = f"bright_{row['filename']}"
    dark_name = f"dark_{row['filename']}"

    cv2.imwrite(str(IMAGE_DIR / bright_name), bright)
    cv2.imwrite(str(IMAGE_DIR / dark_name), dark)

    records.append([bright_name, x1, y1, x2, y2])
    records.append([dark_name, x1, y1, x2, y2])

df_aug = pd.DataFrame(records,columns=["filename", "x_min", "y_min", "x_max", "y_max"])

new_df = pd.concat([df, df_aug], ignore_index=True)
images = []
boxes = []

for josh, row in new_df.iterrows():
    image_path = DATA_DIR / "blank_images" / row["filename"]
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load {row['filename']}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    box = np.array([row["x_min"] / width, row["y_min"] / height, row["x_max"]/width, row["y_max"]/height], dtype=np.float32)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0

    images.append(image)
    boxes.append(box)

X = np.array(images, dtype=np.float32)
y = np.array(boxes, dtype=np.float32)

np.save(DATA_DIR / "blank_X.npy", X)
np.save(DATA_DIR / "blank_y.npy", y)

print("Image data shape:", X.shape)
print("Bounding-box data shape:", y.shape)
print("Pixel range:", X.min(), "to", X.max())
print("Example bounding box:", y[0])
