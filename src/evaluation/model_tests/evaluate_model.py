import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from matplotlib.patches import Rectangle

predictions = np.load("dataset/test_predictions.npy")
labels = pd.read_csv("dataset/test_labels.csv")
CONFIDENCE_THRESHOLD = 0.5

if predictions.ndim != 3 or predictions.shape[1] != 5 or predictions.shape[2] < 6:
    raise ValueError(
        "Prediction data does not include the confidence model. "
        "Run get_prediction_values.py first to regenerate dataset/test_predictions.npy."
    )

for i in range(labels.shape[0]):
    filename = labels.iloc[i]["filename"]
    img = cv2.imread(f"dataset/test_images/{filename}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt, dot, dot_aug, box, confidence_box = predictions[i]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img)
    axes[0].scatter(dot[0], dot[1], color="red", s=80, label="Prediction")
    axes[0].scatter((gt[0] + gt[2]) / 2,(gt[1] + gt[3]) / 2,color="lime", marker="x", s=80, label="True center")
    axes[0].set_title("Dot Model")
    axes[0].legend()

    axes[1].imshow(img)
    axes[1].scatter(dot_aug[0], dot_aug[1], color="red", s=80, label="Prediction")
    axes[1].scatter((gt[0] + gt[2]) / 2,(gt[1] + gt[3]) / 2,color="lime", marker="x", s=80, label="True center")
    axes[1].set_title("Dot Model + Augmentation")
    axes[1].legend()

    axes[2].imshow(img)
    axes[2].add_patch(Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],fill=False, edgecolor="lime", linewidth=2, label="True box"))
    axes[2].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],fill=False, edgecolor="red", linewidth=2, label="Prediction"))
    axes[2].set_title("Box Model + Augmentation")
    axes[2].legend()

    axes[3].imshow(img)
    axes[3].add_patch(Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],fill=False, edgecolor="lime", linewidth=2, label="True box"))
    if confidence_box[5] >= CONFIDENCE_THRESHOLD:
        axes[3].add_patch(Rectangle((confidence_box[0], confidence_box[1]), confidence_box[2] - confidence_box[0],confidence_box[3] - confidence_box[1],fill=False, edgecolor="red", linewidth=2, label="Prediction"))
        result = "face detected"
    else:
        result = "below threshold"
    axes[3].set_title(
        f"Confidence Model: {confidence_box[5]:.2f} ({result})\n"
        f"Inference: {confidence_box[4] * 1000:.1f} ms"
    )
    axes[3].text(
        0.02,
        0.02,
        f"Detection threshold: {CONFIDENCE_THRESHOLD:.2f}",
        transform=axes[3].transAxes,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )
    axes[3].legend()

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.waitforbuttonpress(0)
    plt.close(fig)
