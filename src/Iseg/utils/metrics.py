import tensorflow as tf
import numpy as np


class Metrics:
    def __init__(self):
        self.smooth = 1e-15

    def iou(self, y_true, y_pred):
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)

    
    def dice_coef(self, y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection  + self.smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)

    def dice_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)