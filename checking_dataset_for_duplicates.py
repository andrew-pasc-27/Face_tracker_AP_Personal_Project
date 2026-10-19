
import pandas as pd

df = pd.read_csv("dataset/labels.csv")

name = ""

for index, row in df.iterrows():
    if row["filename"] != name:
        name = row["filename"]
        continue
    else:
        print("Duplicate filename found: " + str(name))
        df.drop(index, inplace=True)

df.to_csv("dataset/labels.csv", index=False)