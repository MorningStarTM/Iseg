import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import re
import numpy as np


class Pipeline:
    def __init__(self, target_size:tuple, path:None, split:int):
        self.target_size = target_size
        self.path = path
        self.split = split

    def load_dataset(self):
        """
        Function for Load the dataset
        dataset folder should have image folder and mask folder

        Return 
            images path(List)
            mask path(list)
        """

        images = sorted(glob(os.path.join(self.path, "images/*")), key=lambda x: int(re.search(r'\d+', x).group()))
        masks = sorted(glob(os.path.join(self.path, "mask/*")), key=lambda x: int(re.search(r'\d+', x).group()))

        return images, masks

    
    def read_img(self, path):
        """
        This function will read, resize and normalize the image

        Args:
            path: image's path

        Return 
            image (array)
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.target_size)
        img = img / 255
        img = img.astype(np.float32)
        return img 
    

    def read_mask(path):
        """
        This function will read, resize and normalize the mask

        Args:
            path: mask's path

        Return 
            Mask (array)
        """

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = mask / 255
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(np.float32)
        return mask
    

    def preprocess(self, x, y):
        """
        This function is used to enable read_img and read_mask function to compatible with tensorflow library. mentioned funtions are used CV2 for read and preprocess it.

        Args:
            x (str) : path of image
            y (str) : path of mask

        Return:
            image (array)
            Mask (array)
        """
        def f(x, y):
            x = x.decode()
            y = y.decode()

            x = self.read_img(x)
            y = self.read_mask(y)
            return x, y

        image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        image.set_shape([256, 256, 3])
        mask.set_shape([256, 256, 1])

        return image, mask



    def tf_dataset(self, x, y, batch=8):
        """
        This function is used to create tensorflow dataset pipeline

        Args:
            x (str) : path of image
            y (str) : path of mask

        Return :
            dataset (tensorflow dataset)
        
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(self.preprocess)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(2)
        return dataset