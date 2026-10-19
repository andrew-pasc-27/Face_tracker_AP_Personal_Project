import numpy as np 
from pathlib import Path
import pandas as pd

DATA_DIR = Path("dataset")
df = pd.read_csv(DATA_DIR / "labels_cleaned.csv")

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

print(df.iloc[128])
