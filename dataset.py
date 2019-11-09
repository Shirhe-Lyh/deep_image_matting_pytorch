# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:46:51 2019

@author: shirhe-lyh
"""

import cv2
import numpy as np
import os
import PIL
import torch
import torchvision as tv

from data import utils


def random_dilate(alpha, low=3, high=30, mode='constant'):
    """Dilation."""
    erode_ksize = np.random.randint(low=low, high=high)
    dilate_ksize = np.random.randint(low=low, high=high)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    alpha_eroded = cv2.erode(alpha, erode_kernel)
    alpha_dilated = cv2.dilate(alpha, dilate_kernel)
    if mode == 'constant':
        value = 128
    else:
        value = np.random.randint(low=100, high=255)
    alpha_noise = value * ((alpha_dilated - alpha_eroded) / 255.)
    alpha_noise += alpha_eroded
    return alpha_noise


class MattingDataset(torch.utils.data.Dataset):
    """Read dataset for Matting."""
    
    def __init__(self, annotation_path, root_dir=None, transforms=None,
                 output_size=320, dilation_mode='constant'):
        self._transforms = transforms
        self._output_size = output_size
        self._dilation_mode = dilation_mode
        
        # Transform
        if transforms is None:
            channel_means = [0.485, 0.456, 0.406]
            channel_std = [0.229, 0.224, 0.225]
            self._transforms = tv.transforms.Compose([
                tv.transforms.ColorJitter(brightness=32/255., contrast=0.5, 
                                      saturation=0.5, hue=0.2),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=channel_means, std=channel_std)])
        
        # Format [[image_path, alpha_path], ...]
        self._image_alpha_paths = self.get_image_mask_paths(annotation_path,
                                                            root_dir=root_dir)
        self._remove_invalid_data()
        
    def __getitem__(self, index):
        image_path, alpha_path = self._image_alpha_paths[index]
        image = PIL.Image.open(image_path)
        alpha = PIL.Image.open(alpha_path)
        
        # Rotate
#        degree = np.random.randint(low=-30, high=30)
#        image = image.rotate(degree)
#        alpha = alpha.rotate(degree)
        # Crop
        width, height = alpha.size
        min_size = np.min((width, height))
        crop_sizes = [320, 480, 600, 800]
        crop_size = np.random.choice(crop_sizes)
        if min_size >= crop_size:
            height_offset = np.random.randint(low=0, high=height-crop_size+1)
            width_offset = np.random.randint(low=0, high=width-crop_size+1)
            box = (width_offset, height_offset, width_offset+crop_size,
                   height_offset+crop_size)
            image = image.crop(box=box)
            alpha = alpha.crop(box=box)
        # Resize
        if crop_size > self._output_size:
            image = image.resize((self._output_size, self._output_size),
                                 PIL.Image.ANTIALIAS)
            alpha = alpha.resize((self._output_size, self._output_size),
                                 PIL.Image.NEAREST)
        # Flip
        prob = np.random.uniform()
        if prob > 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            alpha = alpha.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        # Dilate, Erode
        alpha = np.array(alpha)
        alpha_noise = random_dilate(alpha, mode=self._dilation_mode)
        
        alpha = torch.Tensor(alpha / 255.)
        alpha_noise = torch.Tensor(alpha_noise / 255.)
        # Transform
        image_preprocessed = self._transforms(image)
        alpha_noise_u = torch.unsqueeze(alpha_noise, dim=0)
        image_concated = torch.cat([image_preprocessed, alpha_noise_u], dim=0)
        return image_concated, torch.unsqueeze(alpha, 0), alpha_noise_u
    
    def __len__(self):
        return len(self._image_alpha_paths)
    
    def get_image_mask_paths(self, annotation_path, root_dir=None):
        """Get the paths of images and masks.
        
        Args:
            annotation_path: A file contains the paths of images and masks.
            
        Returns:
            A list [[image_path, mask_path], [image_path, mask_path], ...].
            
        Raises:
            ValueError: If annotation_file does not exist.
        """
        # Format: [[image_path, matting_path, alpha_path, mask_path], ...]
        image_matting_alpha_mask_paths = utils.provide(annotation_path)
        # Remove matting_paths, mask_paths
        image_alpha_paths = []
        for image_path, _, alpha_path, _ in image_matting_alpha_mask_paths:
            if root_dir is not None:
                if not image_path.startswith(root_dir):
                    image_path = os.path.join(root_dir, image_path)
                    alpha_path = os.path.join(root_dir, alpha_path)
                    image_path = image_path.replace('\\', '/')
                    alpha_path = alpha_path.replace('\\', '/')
            image_alpha_paths.append([image_path, alpha_path])
        return image_alpha_paths
            
    def _remove_invalid_data(self):
        valid_data = []
        for image_path, alpha_path in self._image_alpha_paths:
            if not os.path.exists(image_path):
                continue
            if not os.path.exists(alpha_path):
                continue
            valid_data.append([image_path, alpha_path])
        self._image_alpha_paths = valid_data
        