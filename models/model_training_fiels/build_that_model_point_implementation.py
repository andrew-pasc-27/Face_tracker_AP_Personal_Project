from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape

DATA_DIR = Path("dataset")

X = np.load(DATA_DIR / "combined_labeled_X.npy")
y = np.load(DATA_DIR / "combined_labeled_y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def build_model():
    model = Sequential()

    model.add(Input(shape=X_train.shape[1:]))

    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(2, activation="sigmoid"))
    return model

model = build_model()
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16)

model.summary()
model.save("labeled_face_turret_model.keras")
