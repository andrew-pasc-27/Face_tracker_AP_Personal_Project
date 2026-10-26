# Face_tracker_AP_Personal_Project
A face-tracking turret built with a custom-trained convolutional neural network (CNN), Python, and Arduino-controlled servos. The goal is to detect a face through a webcam and move a pan-tilt mount to follow it.

## Project Overview

I built and trained my own CNN to locate my face in webcam images rather than relying entirely on a pretrained face detector. I experimented with different ways of representing a face's location and investigated whether data augmentation could improve performance on images the model had not seen before.

## Model Experiments

- **Dot Model:** Predicts the center of a face.
- **Box Model:** Predicts the bounding box surrounding a face.
- **Augmented Box Model:** Uses augmented training images to test whether greater variation improves detection.
- **Model with Confidence:** Adds a fifth output, a face confidence score from 0.0 to 1.0, using new images without a face.

## Testing and Evaluation

Evaluation has been orchestrated through [evaluate_model.py](https://github.com/andrew-pasc-27/Face_tracker_AP_Personal_Project/blob/main/src/evaluation/model_tests/evaluate_model.py). 17 new images were tested against the dot model, box model, augmented box model, and the model with confidence. 
Average pixel error from the center across all 17 test images:
- Dot model: 221.06 px
- Augmented dot model: 224.92 px
- Augmented box model: 67.94 px
- Confidence model: 126.39 px; average confidence 0.9129

Surprisingly, the augmented box model nearly achieved nearly half the error of the confidence model. Even though there was a face in each picture, the model with confidence only had an average confidence of 0.91, likely attributable to the location of my face in each of the test images. The dot model and the augmented dot model see minimal changes in error after augmentation, since both mostly guess the center. 

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
