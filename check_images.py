import cv2
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
df = pd.read_csv(DATA_DIR / "labels_cleaned.csv")

# Display a sample image with its bounding box
row = df.sample(1).iloc[0]

image = cv2.imread(str(DATA_DIR / "images" / row["filename"]))

x1, y1, x2, y2 = map(
    int, [row["x1"], row["y1"], row["x2"], row["y2"]]
)

cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imshow("Face Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()