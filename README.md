# Face_tracker_AP_Personal_Project
A face-tracking gimbal built with a custom-trained convolutional neural network (CNN), Python, and Arduino-controlled servos. The goal is to detect a face through a webcam and move a pan-tilt mount to follow it.

## Project Overview

I built and trained my own CNN to locate my face in webcam images to be used in a fake-tracking gimbal. I experimented with different ways of representing a face's location and investigated whether data augmentation could improve performance on images the model had not seen before.

## Methodology

### 1. Custom CNN Development
Using the TensorFlow / Keras library, I explored different ways to represent the target location of the center of my face. I used mean squared error for my loss function, and sigmoid for my activation functions. 

#### **Dot Model** 
I first trained the model with images annotated with a single point at the center of my face; this implementation showed poor results, as the model simply memorized the location of the dot as near the center of the screen, not conflating it with the center of my face.

#### **Bounding Box Model**
Representing the face as a region instead of a single point allows the model to recognize more patterns around my face, which would likely improve the model’s performance. I tested a custom implementation of Generalized Intersection over Union (the built in functions have been deprecated), but it predicted the whole webcam frame as my face.
<br>
The model returns 4 coordinates:
- x1
- y1
- x2
- y2
<br>
These points can be translated into the coordinates (x1, y1), (x2, y2) 
Annotation was implemented through OpenCV’s `selectROI()` method. Issues likely arise in the consistency and accuracy in my annotations. However, the webcam gimbal often hallucinates a face, and I introduced fixes for that below.
<br>
### 2. Data Augmentation
I experimented with data augmentation to see whether modified images would increase the accuracy of the model on unseen images. 
The image augmentations were implemented through OpenCV’s ‘flip()` and `convertScaleAbs()`. The images were flipped horizontally, flipped vertically, and modified in brightness for 4 augmentations per image.
<br>
For the dot model, augmentation likely has a very small effect due to its memorization of the general location of the face center; the augmented images have similar centers. However, when tested on the box model, augmentation improved accuracy significantly.
<br>
### 3. Confidence Model
I also tested a model with five outputs: four bounding-box coordinates and a confidence score between 0 and 1. Because the webcam output might not always include a face, I introduced blank images to the model annotated with a 0 confidence score, which would hopefully stop the original box model from hallucinating a face.

### 4. Testing and Evaluation

Evaluation has been orchestrated through [evaluate_model.py](https://github.com/andrew-pasc-27/Face_tracker_AP_Personal_Project/blob/main/src/evaluation/model_tests/evaluate_model.py). 17 new images were tested against the dot model, box model, augmented box model, and the model with confidence. 
<br>
Average pixel error from the center across all 17 test images:
- Dot model: 221.06 px
- Augmented dot model: 224.92 px
- Augmented box model: 67.94 px
- Confidence model: 126.39 px; average confidence 0.9129
<br>
Surprisingly, the augmented box model nearly achieved nearly half the error of the confidence model. Even though there was a face in each picture, the model with confidence only had an average confidence of 0.91, likely attributable to the location of my face in each of the test images. The dot model and the augmented dot model see minimal changes in error after augmentation, since both mostly guess the center. 
<br>
**Limitations**<br>
The results are based on 17 test images. Although these images were new to the models, this is a relatively small evaluation set, so the measured errors may not represent performance across all lighting conditions, backgrounds, face positions, or camera distances.

The confidence experiment also used test images that all contained a face. Therefore, the average confidence of 0.9129 does not establish how reliably the model distinguishes images with faces from images without them.

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

Real-time tracking improvement is still in progress.
