
import cv2
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_DIR = DATA_DIR / "images"
AUG_DIR = DATA_DIR / "augmented_images"
AUG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "labels.csv")
records = []

for john_cena_du_du_du_du, row in df.iterrows():
    image_path = IMAGE_DIR / row["filename"]
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"could not load {row['filename']}")
        continue

    h, w = image.shape[:2]
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

    flipped = cv2.flip(image, 1)
    new_x1 = w - 1 - x2
    new_x2 = w - 1 - x1


    cv2.imwrite(str(AUG_DIR / f"lr_{row['filename']}"), flipped)

    records.append([f"lr_{row['filename']}", new_x1, y1, new_x2, y2])

    flipped = cv2.flip(image, 0)
    new_y1 = h - 1 - y2
    new_y2 = h - 1 - y1

    cv2.imwrite(str(AUG_DIR / f"ud_{row['filename']}"), flipped)

    records.append([f"ud_{row['filename']}", x1, new_y1, x2, new_y2])

    bright = cv2.convertScaleAbs(image, alpha=1.0, beta=40)

    dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-40)

    bright_name = f"bright_{row['filename']}"
    dark_name = f"dark_{row['filename']}"

    cv2.imwrite(str(AUG_DIR / bright_name), bright)
    cv2.imwrite(str(AUG_DIR / dark_name), dark)

    records.append([bright_name, x1, y1, x2, y2])
    records.append([dark_name, x1, y1, x2, y2])

df_aug = pd.DataFrame(records,columns=["filename", "x1", "y1", "x2", "y2"])

df_aug.to_csv(DATA_DIR / "augmented_labels.csv", index=False)

print(f"saved {len(df_aug)} augmented images and labels.")
