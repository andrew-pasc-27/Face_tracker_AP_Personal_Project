    
# Generalized Intersection over Union (GIoU) loss
# Implementation adapted with ChatGPT assistance.
# Based on the GIoU method described in:
# Rezatofighi et al., "Generalized Intersection
# over Union: A Metric and A Loss for Bounding
# Box Regression" (CVPR 2019).
# https://arxiv.org/abs/1902.09630

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="CustomLoss")
class GIoULoss(tf.keras.losses.Loss):
    def __init__(self, name="stable_giou_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # 1. Cast and extract coordinates.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        true_x1 = tf.minimum(y_true[:, 0], y_true[:, 2])
        true_y1 = tf.minimum(y_true[:, 1], y_true[:, 3])
        true_x2 = tf.maximum(y_true[:, 0], y_true[:, 2])
        true_y2 = tf.maximum(y_true[:, 1], y_true[:, 3])

        pred_x1 = tf.minimum(y_pred[:, 0], y_pred[:, 2])
        pred_y1 = tf.minimum(y_pred[:, 1], y_pred[:, 3])
        pred_x2 = tf.maximum(y_pred[:, 0], y_pred[:, 2])
        pred_y2 = tf.maximum(y_pred[:, 1], y_pred[:, 3])

        # 2. Calculate individual box areas. Both boxes are now valid even
        # when the network temporarily predicts x2 < x1 or y2 < y1.
        area1 = (true_x2 - true_x1) * (true_y2 - true_y1)
        area2 = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        
        # 3. Intersection coordinates & area
        inter_x1 = tf.maximum(true_x1, pred_x1)
        inter_y1 = tf.maximum(true_y1, pred_y1)
        inter_x2 = tf.minimum(true_x2, pred_x2)
        inter_y2 = tf.minimum(true_y2, pred_y2)
        
        inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
        inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        
        # 4. Union area (with small epsilon to prevent Division by Zero)
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        union = tf.maximum(area1 + area2 - intersection, 0.0) + epsilon
        iou = intersection / union
        
        # 5. Enclosing Box (Convex Hull) coordinates & area
        enclose_x1 = tf.minimum(true_x1, pred_x1)
        enclose_y1 = tf.minimum(true_y1, pred_y1)
        enclose_x2 = tf.maximum(true_x2, pred_x2)
        enclose_y2 = tf.maximum(true_y2, pred_y2)
        
        enclose_w = tf.maximum(0.0, enclose_x2 - enclose_x1)
        enclose_h = tf.maximum(0.0, enclose_y2 - enclose_y1)
        enclose_area = (enclose_w * enclose_h) + epsilon
        
        # 6. Calculate GIoU
        giou = iou - ((enclose_area - union) / enclose_area)
        
        # Bound GIoU between -1.0 and 1.0 just in case of rounding errors
        giou = tf.clip_by_value(giou, -1.0, 1.0)
        
        # Return loss: minimizes when GIoU approaches 1.0
        return 1.0 - giou

    def get_config(self):
        config = super().get_config()
        return config


