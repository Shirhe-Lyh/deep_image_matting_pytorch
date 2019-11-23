# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:43:38 2019

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os
import uuid

import retrieve


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
    root_dir = 'xxx/Combined_Dataset'
    bg_image_root_dir = '/data/COCO/train2017'
    output_dir = '/data/matting/dim_composite_images'
    train_txt_path = './train.txt'
    val_txt_path = './val.txt'
    num_bg_images_per_fg = 50
    
    train_fg_alpha_paths = retrieve.get_image_paths(root_dir)
    test_fg_alpha_paths = retrieve.get_image_paths(root_dir, dataset='Test_set')
    
    bg_image_paths = glob.glob(os.path.join(bg_image_root_dir, '*.*'))
    np.random.shuffle(bg_image_paths)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate training data
    index = 0
    fg_bg_alpha_paths = []
    iterator = iter(bg_image_paths)
    for fg_path, alpha_path in train_fg_alpha_paths:
        for i in range(num_bg_images_per_fg):
            fg = cv2.imread(fg_path)
            alpha = cv2.imread(alpha_path, 0)
            bg_path = next(iterator)
            bg = cv2.imread(bg_path)
            image = compose(fg, bg, alpha)
            if image is None:
                print(fg_path)
                continue
            output_path = os.path.join(output_dir, str(uuid.uuid4()) + '.jpg')
            cv2.imwrite(output_path, image)
            
            bg_path = bg_path.replace('\\', '/')
            fg_bg_alpha_paths.append([output_path, fg_path, alpha_path, bg_path])
        index += 1  
        if index % 50 == 0:
            print('On image: {}/{}'.format(index, len(train_fg_alpha_paths)))
        
    retrieve.write_to_txt(fg_bg_alpha_paths, train_txt_path)
        
    # Generate validation data
    fg_bg_alpha_paths = []
    num_bg_images_per_fg = 10
    iterator = iter(bg_image_paths)
    for fg_path, alpha_path in test_fg_alpha_paths:
        for i in range(num_bg_images_per_fg):
            bg_path = next(iterator)
            bg_path = bg_path.replace('\\', '/')
            fg_bg_alpha_paths.append([fg_path, alpha_path, bg_path])
        
    retrieve.write_to_txt(fg_bg_alpha_paths, val_txt_path)

    
    

