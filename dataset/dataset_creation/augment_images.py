
import cv2
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_DIR = DATA_DIR / "images"
AUG_DIR = DATA_DIR / "augmented_images"
AUG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "points.csv")
records = []

for john_cena_du_du_du_du, row in df.iterrows():
    image_path = IMAGE_DIR / row["filename"]
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load {row['filename']}")
        continue

    h, w = image.shape[:2]
    x, y = row["center_x"], row["center_y"]

    flipped = cv2.flip(image, 1)
    new_x = w - 1 - x

    cv2.imwrite(str(AUG_DIR / f"lr_{row['filename']}"), flipped)

    records.append([f"lr_{row['filename']}", new_x, y])

    flipped = cv2.flip(image, 0)
    new_y = h - 1 - y

    cv2.imwrite(str(AUG_DIR / f"ud_{row['filename']}"), flipped)

    records.append([f"ud_{row['filename']}", x, new_y])

    bright = cv2.convertScaleAbs(image, alpha=1.0, beta=40)

    dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-40)

    bright_name = f"bright_{row['filename']}"
    dark_name = f"dark_{row['filename']}"

    cv2.imwrite(str(AUG_DIR / bright_name), bright)
    cv2.imwrite(str(AUG_DIR / dark_name), dark)

    records.append([bright_name, x, y])
    records.append([dark_name, x, y]) 

df_aug = pd.DataFrame(records,columns=["filename", "center_x", "center_y"])

df_aug.to_csv(DATA_DIR / "augmented_points.csv", index=False)

print(f"Saved {len(df_aug)} augmented images and labels.")
