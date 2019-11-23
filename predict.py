# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:50:00 2019

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os
import torch
import torchvision as tv

import dataset
import model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def composite(image, alpha):
    alpha_exp = np.expand_dims(alpha, axis=2)
    image_ = np.concatenate([image, alpha_exp], axis=2)
    return image_


if __name__ == '__main__':
    ckpt_path = './models/model.ckpt'
    test_images_dir = './test/'
    output_masks_dir = './test/pred_alphas'
    test_images_paths = glob.glob(os.path.join(test_images_dir, '*.png'))
    
    if not os.path.exists(ckpt_path):
        raise ValueError('`ckpt_path` does not exist.')
    if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)
    
    feature_extractor = model.vgg16_bn_feature_extractor(
        model.VGG16_BN_CONFIGS.get('13conv')).to(device)
    dim = model.DIM(feature_extractor).to(device)
    #dim.load_state_dict(torch.load(ckpt_path))
    dim_pretrained_params = torch.load(ckpt_path).items()
    dim_state_dict = {k.replace('module.', ''): v for k, v in
                      dim_pretrained_params}
    dim.load_state_dict(dim_state_dict)
    print('Load DIM pretrained parameters, Done')
    
    # Transform
    channel_means = [0.485, 0.456, 0.406]
    channel_std = [0.229, 0.224, 0.225]
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=channel_means, std=channel_std)])
    
    dim.eval()
    with torch.no_grad():
        for image_path in test_images_paths:
            image = cv2.imread(image_path, -1)
            image_fg = cv2.imread(image_path.replace('.png', '.jpg'))
            alpha = image[:, :, 3]
            image_rgb = cv2.cvtColor(image_fg, cv2.COLOR_BGR2RGB)
            image_processed = transforms(image_rgb).to(device)
            
            alpha_noise = dataset.random_dilate(alpha)
            alpha_noise_exp = np.expand_dims(alpha_noise / 255., axis=0)
            alpha_noise_exp = torch.Tensor(alpha_noise_exp).to(device)
            images = torch.cat([image_processed, alpha_noise_exp], dim=0)
            images = torch.unsqueeze(images, dim=0)
            
            outputs = dim(images)
            alpha_pred = outputs.data.cpu().numpy()[0][0]
            alpha_pred = 255 * alpha_pred
            alpha_pred = alpha_pred.astype(np.uint8)
            
            image_name = image_path.split('/')[-1]
            output_path = os.path.join(output_masks_dir, image_name)
            cv2.imwrite(output_path, composite(image_fg, alpha_pred))
            output_path = os.path.join(output_masks_dir, 
                                       image_name.replace('.png', '_alpha.png'))
            cv2.imwrite(output_path, alpha)
            output_path = os.path.join(output_masks_dir, 
                                       image_name.replace('.png', '_matte.png'))
            cv2.imwrite(output_path, alpha_pred)
            output_path = os.path.join(output_masks_dir, 
                                       image_name.replace('.png', '_noise.png'))
            cv2.imwrite(output_path, alpha_noise)
        