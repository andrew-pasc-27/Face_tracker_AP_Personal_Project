import numpy as np 
from pathlib import Path

DATA_DIR = Path("dataset")

x1 = np.load(DATA_DIR / "aug_X.npy")
y1 = np.load(DATA_DIR / "aug_y.npy")
x2 = np.load(DATA_DIR / "X.npy")
y2 = np.load(DATA_DIR / "y.npy")

if x1.shape[1:] == x2.shape[1:] and y1.shape[1:] == y2.shape[1:]:
    X = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    np.save(DATA_DIR / "combined_X.npy", X)
    np.save(DATA_DIR / "combined_y.npy", y)
    print("Combined data shape:", X.shape, y.shape)