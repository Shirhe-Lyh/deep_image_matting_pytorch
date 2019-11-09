# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:44:07 2018

@author: shirhe-lyh
"""

import cv2
import numpy as np
import os


def get_alpha(image):
    """Returns the alpha channel of a given image."""
    if image.shape[2] > 3:
        alpha = image[:, :, 3]
        #alpha = remove_noise(alpha)
    else:
        reduced_image = np.sum(np.abs(255 - image), axis=2)
        alpha = np.where(reduced_image > 100, 255, 0)
    alpha = alpha.astype(np.uint8)
    return alpha


def remove_noise(gray, area_threshold=5000):
    gray = gray.astype(np.uint8)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    remove_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_threshold:
            remove_contours.append(contour)
    
    cv2.fillPoly(gray, remove_contours, 0)
    return gray


def to_mask(alpha, threshold=50):
    mask = np.where(alpha > threshold, 1, 0)
    return mask.astype(np.uint8)


def provide(txt_path, delimiter='@'):
    """Returns the paths of images.
    
    Args:
        txt_path: A .txt file with format:
            [image_path_11, image_path_12, ..., image_path_1n,
             image_path_21, image_path_22, ..., image_path_2n,
             ...].
        
    Returns:
        The paths of images.
        
    Raises:
        ValueError: If txt_path does not exist.
    """
    if not os.path.exists(txt_path):
        raise ValueError('`txt_path` does not exist.')
        
    with open(txt_path, 'r') as reader:
        txt_content = np.loadtxt(reader, str, delimiter=delimiter)
        np.random.shuffle(txt_content)
    image_paths = []
    for line in txt_content:
        paths = [x for x in line]
        image_paths.append(paths)
    return image_paths
