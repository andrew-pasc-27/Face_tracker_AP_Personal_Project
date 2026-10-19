
# Generalized Intersection over Union (GIoU) loss
# Implementation adapted with ChatGPT assistance.
# Based on the GIoU method described in:
# Rezatofighi et al., "Generalized Intersection
# over Union: A Metric and A Loss for Bounding
# Box Regression" (CVPR 2019).
# https://arxiv.org/abs/1902.09630

import tensorflow as tf


def giou_loss(y_true, y_pred):
    px1 = tf.minimum(y_pred[..., 0], y_pred[..., 2])
    py1 = tf.minimum(y_pred[..., 1], y_pred[..., 3])
    px2 = tf.maximum(y_pred[..., 0], y_pred[..., 2])
    py2 = tf.maximum(y_pred[..., 1], y_pred[..., 3])

    tx1, ty1, tx2, ty2 = tf.unstack(y_true, axis=-1)

    ix1 = tf.maximum(px1, tx1)
    iy1 = tf.maximum(py1, ty1)
    ix2 = tf.minimum(px2, tx2)
    iy2 = tf.minimum(py2, ty2)

    intersection = (
        tf.maximum(ix2 - ix1, 0.0) *
        tf.maximum(iy2 - iy1, 0.0)
    )

    pred_area = (px2 - px1) * (py2 - py1)
    true_area = (
        tf.maximum(tx2 - tx1, 0.0) *
        tf.maximum(ty2 - ty1, 0.0)
    )

    union = pred_area + true_area - intersection
    iou = intersection / (union + 1e-7)

    cx1 = tf.minimum(px1, tx1)
    cy1 = tf.minimum(py1, ty1)
    cx2 = tf.maximum(px2, tx2)
    cy2 = tf.maximum(py2, ty2)

    enclosing_area = (cx2 - cx1) * (cy2 - cy1)

    giou = iou - (
        (enclosing_area - union) /
        (enclosing_area + 1e-7)
    )

    return 1.0 - giou
