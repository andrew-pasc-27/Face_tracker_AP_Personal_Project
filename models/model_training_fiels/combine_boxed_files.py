import numpy as np
import pandas as pd

x1 = np.load("dataset/X.npy")
x2 = np.load("dataset/aug_X.npy")
y1 = np.load("dataset/y_labels.npy")
y2 = np.load("dataset/aug_y_labels.npy")

combined_labeled_x = np.concatenate((x1, x2), axis=0)
combined_labeled_y = np.concatenate((y1, y2), axis=0)

np.save("dataset/combined_labeled_X.npy", combined_labeled_x)
np.save("dataset/combined_labeled_y.npy", combined_labeled_y)

df = pd.DataFrame(combined_labeled_y, columns=["x1", "y1", "x2", "y2"])
df.to_csv("dataset/combined_labeled_y_labels.csv", index=False)
