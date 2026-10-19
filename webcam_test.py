
import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model("labeled_face_turret_model.keras")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("unable to open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("unable to read webcam frame")
        break

    height, width = frame.shape[:2]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    x1 = int(np.clip(prediction[0], 0, 1) * width)
    y1 = int(np.clip(prediction[1], 0, 1) * height)
    x2 = int(np.clip(prediction[2], 0, 1) * width)
    y2 = int(np.clip(prediction[3], 0, 1) * height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame,f"Predicted: ({x1}, {y1}), ({x2}, {y2})",(10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0),2
    )

    cv2.imshow("Face Center Tracking Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
