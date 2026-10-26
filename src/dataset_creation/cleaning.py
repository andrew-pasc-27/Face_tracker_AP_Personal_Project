
import pandas as pd
from pathlib import Path

# Load dataset
DATA_DIR = Path("dataset")
df = pd.read_csv(DATA_DIR / "labels.csv")

# Keep only the columns containing your actual annotations
df = df[["filename", "x1", "y1", "x2", "y2"]].copy()

# Find images with no bounding box
zero_box = (
    (df["x1"] == 0) &
    (df["y1"] == 0) &
    (df["x2"] == 0) &
    (df["y2"] == 0)
)

print("Unannotated images:")
print(df.loc[zero_box, "filename"])

# Remove unannotated images
df = df.loc[~zero_box].copy()

# Check for invalid bounding boxes
invalid = (
    (df["x2"] <= df["x1"]) |
    (df["y2"] <= df["y1"])
)

print("\nInvalid bounding boxes:", invalid.sum())

if invalid.any():
    print(df.loc[invalid])
else:
    print("All bounding boxes have positive width and height!")

# Save cleaned dataset without overwriting the original
df.to_csv(DATA_DIR / "labels_cleaned.csv", index=False)

print(f"\nAnnotated images: {len(df)}")
print("\nFirst five rows:")
print(df.head())
