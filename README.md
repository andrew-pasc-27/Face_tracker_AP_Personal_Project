# Face_tracker_AP_Personal_Project
A face-tracking turret built with a custom-trained convolutional neural network (CNN), Python, and Arduino-controlled servos. The goal is to detect a face through a webcam and move a pan-tilt mount to follow it.

## Project Overview

I built and trained my own CNN to locate my face in webcam images rather than relying entirely on a pretrained face detector. I experimented with different ways of representing a face's location and investigated whether data augmentation could improve performance on images the model had not seen before.

## Model Experiments

- **Dot Model:** Predicts the center of a face.
- **Box Model:** Predicts the bounding box surrounding a face.
- **Augmented Box Model:** Uses augmented training images to test whether greater variation improves detection.

## Testing and Evaluation

I plan to evaluate the models on a separate set of 50 images that are not used during training. The comparison will focus on prediction accuracy, inference speed, and consistency when tracking a moving face.

## Hardware and Tools

- Webcam
- Arduino and pan-tilt servos
- Python
- TensorFlow / Keras
- OpenCV
- NumPy and pandas

## Goals

- Compare different approaches to face localization.
- Investigate how data augmentation affects performance on unseen images.
- Improve the stability and responsiveness of real-time face tracking.
- Connect the vision model to a physical tracking system.

## Project Status

In progress. Model evaluation and improvements to real-time tracking are ongoing.