import pandas as pd
import cv2
from pathlib import Path

DATA_DIR = Path("dataset")
IMAGE_DIR = DATA_DIR / "images"

df = pd.read_csv(DATA_DIR / "points.csv")
model = cv2.FaceDetectorYN_create("/Users/dr/face-turret/face_detection_yunet_2023mar.onnx", config="", input_size=(960, 1280), score_threshold=0.6, nms_threshold=0.3, top_k=5000)

print(df.head())
records = []

for index, row in df.iterrows():
    filename = row["filename"]
    frame = cv2.imread(str(IMAGE_DIR / filename))
    if frame is not None:
        h, w = frame.shape[:2]
        model.setInputSize((w, h))
        john, faces = model.detect(frame)
        if faces is not None:
            print("skipping annotation for"+ str(filename)+ "because faces were detected")
            for face in faces:
                x, y, box_width, box_height = face[:4]
                records.append([filename, x, y, x+box_width, y+box_height])

        else:
            print(f"Annotating {index}/{len(df)}: {filename}")

            x1, y1, w, h = cv2.selectROI(f"Face Annotation - {index}", frame, False)
            cv2.destroyAllWindows()

            if w == 0 or h == 0:
                print("Skipped image")
                continue

            records.append([filename, x1, y1, x1 + w, y1 + h])            
    else:
        print("could not load" + str(filename))

print("annotated " + str(len(records)) + " images")
annotated_df = pd.DataFrame(records, columns=["filename", "x1", "y1", "x2", "y2"])
annotated_df.to_csv(DATA_DIR / "labels.csv", index=False)