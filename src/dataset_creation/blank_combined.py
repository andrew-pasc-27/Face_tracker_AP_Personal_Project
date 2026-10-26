import numpy as np

X_faces = np.load("dataset/combined_labeled_X.npy")
y_faces = np.load("dataset/combined_labeled_y.npy")

X_empty = np.load("dataset/blank_X.npy")

print("Face images:", X_faces.shape, X_faces.dtype)
print("Face boxes:", y_faces.shape, y_faces.dtype)
print("Empty images:", X_empty.shape, X_empty.dtype)

print("Face pixel range:", X_faces.min(), X_faces.max())
print("Empty pixel range:", X_empty.min(), X_empty.max())


assert X_faces.ndim == 4
assert X_empty.ndim == 4
assert X_faces.shape[1:] == X_empty.shape[1:]
assert y_faces.shape == (len(X_faces), 4)

X_faces = X_faces.astype(np.float32)
X_empty = X_empty.astype(np.float32)

if X_faces.max() > 1.5:
    X_faces /= 255.0

if X_empty.max() > 1.5:
    X_empty /= 255.0

X_all = np.concatenate([X_faces, X_empty], axis=0)

y_bbox_all = np.concatenate([
    y_faces.astype(np.float32),
    np.zeros((len(X_empty), 4), dtype=np.float32)
])

y_conf = np.concatenate([
    np.ones((len(X_faces), 1), dtype=np.float32),
    np.zeros((len(X_empty), 1), dtype=np.float32)
])

bbox_weights = np.concatenate([
    np.ones(len(X_faces), dtype=np.float32),
    np.zeros(len(X_empty), dtype=np.float32)
])

np.save("dataset/face_tracker_X.npy", X_all)
np.save("dataset/face_tracker_bbox.npy", y_bbox_all)
np.save("dataset/face_tracker_conf.npy", y_conf)
np.save("dataset/face_tracker_bbox_weights.npy", bbox_weights)

print("Combined images:", X_all.shape)
print("Bounding boxes:", y_bbox_all.shape)
print("Confidence labels:", y_conf.shape)
print("Bounding-box weights:", bbox_weights.shape)
print("Face images:", int(y_conf.sum()))
print("Empty images:", int((y_conf == 0).sum()))