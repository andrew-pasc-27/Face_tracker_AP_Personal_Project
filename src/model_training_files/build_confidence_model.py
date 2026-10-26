from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from custom_giou_loss import GIoULoss

X = np.load("dataset/face_tracker_X.npy")
y_bbox = np.load("dataset/face_tracker_bbox.npy")
y_conf = np.load("dataset/face_tracker_conf.npy")
bbox_weights = np.load("dataset/face_tracker_bbox_weights.npy")

indices = np.arange(len(X))
train_idx, val_idx = train_test_split(indices,test_size=0.20,shuffle=True,random_state=42)

X_train, X_val = X[train_idx], X[val_idx]
bbox_train, bbox_val = y_bbox[train_idx], y_bbox[val_idx]
conf_train, conf_val = y_conf[train_idx], y_conf[val_idx]
weights_train, weights_val = bbox_weights[train_idx], bbox_weights[val_idx]

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Training empty images:", int(np.sum(conf_train == 0)))
print("Validation empty images:", int(np.sum(conf_val == 0)))

def build_model():
    inputs = Input(shape=X_train.shape[1:])

    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    bbox = Dense(4, activation="sigmoid", name="bbox")(x)
    confidence = Dense(1, activation="sigmoid", name="confidence")(x)

    return Model(inputs=inputs,outputs={"bbox": bbox, "confidence": confidence})

model = build_model()
model.compile(optimizer="adam", loss={"bbox": GIoULoss(), "confidence":"binary_crossentropy"}, loss_weights= {"bbox":1.0, "confidence":1.0}, weighted_metrics={"bbox": ["mae"],"confidence": [keras.metrics.BinaryAccuracy(name="accuracy",threshold=0.5)]})

history = model.fit(
    X_train,{"bbox": bbox_train,"confidence": conf_train},sample_weight={"bbox": weights_train,"confidence": np.ones(len(X_train), dtype=np.float32)},
    validation_data=(X_val,{"bbox": bbox_val,"confidence": conf_val},{"bbox": weights_val,"confidence": np.ones(len(X_val), dtype=np.float32)}
    ),epochs=20,batch_size=16,shuffle=False)

model.summary()
model.save("confidence_face_turret_model.keras")
