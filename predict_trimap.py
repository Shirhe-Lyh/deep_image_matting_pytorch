# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:01:35 2019

@author: john
"""

import cv2
import glob
import numpy as np
import os
import torch
import torchvision as tv

import model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def matte(image, alpha):
    alpha_exp = np.expand_dims(alpha, axis=2)
    image_ = np.concatenate([image, alpha_exp], axis=2)
    return image_


def compose(fg, bg, alpha):
    if fg is None or bg is None or alpha is None:
        return None
    
    height, width, _ = fg.shape
    height_bg, width_bg, _ = bg.shape
    alpha_exp = np.expand_dims(alpha, axis=2) / 255.
    if min(height_bg, width_bg) >= max(height, width):
        bg_resized = bg[:height, :width]
    else:
        bg_resized = cv2.resize(bg, (width, height))
    image = alpha_exp * fg + (1 - alpha_exp) * bg_resized
    return image.astype(np.uint8)


if __name__ == '__main__':
    ckpt_path = './models/model.ckpt'
    test_fg_dir = './data_dim/test/fg'
    test_bg_dir = './data_dim/test/bg'
    test_alpha_dir = './data_dim/test/alpha'
    test_trimap_dir = './data_dim/test/trimap'
    output_dir = './data_dim/test/preds'
    test_fg_paths = glob.glob(os.path.join(test_fg_dir, '*.*'))
    test_bg_paths = glob.glob(os.path.join(test_bg_dir, '*.*'))
    
    if not os.path.exists(ckpt_path):
        raise ValueError('`ckpt_path` does not exist.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    feature_extractor = model.vgg16_bn_feature_extractor(
        model.VGG16_BN_CONFIGS.get('13conv'), pretrained=False).to(device)
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
        for fg_path, bg_path in zip(test_fg_paths, test_bg_paths):
            fg = cv2.imread(fg_path)
            bg = cv2.imread(bg_path)
            image_name = fg_path.replace('\\', '/').split('/')[-1]
            alpha_path = os.path.join(test_alpha_dir, image_name)
            alpha = cv2.imread(alpha_path, 0)
            trimap_name = image_name.replace('.png', '_0.png')
            trimap_path = os.path.join(test_trimap_dir, trimap_name)
            trimap = cv2.imread(trimap_path, 0)
            image = compose(fg, bg, alpha)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_size = image_rgb.shape[:2]
            
            image_processed = transforms(image).to(device)
            trimap_exp = np.expand_dims(trimap / 255., axis=0)
            trimap_exp = torch.Tensor(trimap_exp).to(device)
            images = torch.cat([image_processed, trimap_exp], dim=0)
            images = torch.unsqueeze(images, dim=0)
                
            alphas_pred = dim(images)
            
            alpha_pred_ = alphas_pred.data.cpu().numpy()[0][0]
            alpha_pred_ = 255 * alpha_pred_
            alpha_pred = alpha_pred_.astype(np.uint8)
            alpha_pred = np.where(trimap == 128, alpha_pred, trimap)
            
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, image)
            output_path = os.path.join(output_dir, 
                                       image_name.replace('.png', '_matte.png'))
            cv2.imwrite(output_path, matte(image, alpha_pred))
            output_path = os.path.join(output_dir, 
                                       image_name.replace('.png', '_alpha.png'))
            cv2.imwrite(output_path, alpha_pred)
        