from pathlib import Path
import pandas as pd
import numpy as np

y = np.load("dataset/y.npy")
records = []
counter = 0
for item in sorted(Path("dataset/images").iterdir()):
    if item.is_file() and item.suffix == ".jpg":
        if counter < len(y):
            records.append({"filename": item.name, "center_x": int(y[counter][0] * 1280), "center_y": int(y[counter][1] * 960)})
        else:
            records.append({"filename": item.name, "center_x": 0, "center_y": 0})
    counter += 1

df = pd.DataFrame(records)

df.to_csv("dataset/points.csv", index=False)
print(df.head())