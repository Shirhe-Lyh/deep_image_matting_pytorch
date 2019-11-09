# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:47:28 2019

@author: shirhe-lyh
"""

import cv2
import numpy as np
import os

import utils


def get_image_paths(root_dir):
    image_paths_dict = {}
    matting_paths_dict = {}
    for root, dirs, files in os.walk(root_dir):
        if not files:
            continue
        
        for file in files:
            file_path = os.path.join(root, file)
            file_path = file_path.replace('\\', '/')
            file_name = file.split('.')[0]
            dir_name = file_path.split('/')[-2]
            if dir_name.startswith('clip'):
                image_paths_dict[file_name] = file_path
            if dir_name.startswith('matting'):
                matting_paths_dict[file_name] = file_path
    
    image_corresponding_paths = []
    for image_name, path in image_paths_dict.items():
        matting_path = matting_paths_dict.get(image_name, None)
        if matting_path is not None:
            image_corresponding_paths.append([path, matting_path])
        else:
            print(path)
    print('Number of valid images: ', len(image_corresponding_paths))
    if len(image_corresponding_paths) < 1:
        raise ValueError('`root_dir` is error. Please reset it correctly.')
    return image_corresponding_paths


def split(image_paths, num_val_samples=100):
    if image_paths is None:
        return None
    
    np.random.shuffle(image_paths)
    val_image_paths = image_paths[:num_val_samples]
    train_image_paths = image_paths[num_val_samples:]
    return train_image_paths, val_image_paths


def write_to_txt(image_paths, txt_path, delimiter='@'):
    if image_paths is None:
        return
    
    with open(txt_path, 'w') as writer:
        for element in image_paths:
            line = delimiter.join(element)
            writer.write(line + '\n')
    print('Write successfully to: ', txt_path)
    
    
def write_masks(image_paths, root_dir, add_mask_paths=True):
    if not image_paths:
        return image_paths
    
    alpha_dir = os.path.join(root_dir, 'alphas')
    mask_dir = os.path.join(root_dir, 'masks')
    if not os.path.exists(alpha_dir):
        os.mkdir(alpha_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    
    new_image_paths = []
    for i, [image_path, matting_path] in enumerate(image_paths):
        if (i + 1) % 1000 == 0:
            print('On image: {}/{}'.format(i + 1, len(image_paths)))
        
        matting_image = cv2.imread(matting_path, -1)
        if matting_image is None:
            print('Image does not exist: ', matting_path)
            continue
        alpha = utils.get_alpha(matting_image)
        mask = utils.to_mask(alpha)
        image_name = matting_path.split('/')[-1]
        alpha_path = os.path.join(alpha_dir, image_name)
        alpha_path = alpha_path.replace('\\', '/')
        mask_path = os.path.join(mask_dir, image_name)
        mask_path = mask_path.replace('\\', '/')
        cv2.imwrite(alpha_path, alpha)
        cv2.imwrite(mask_path, mask)
        
        if add_mask_paths:
            new_image_paths.append([image_path, matting_path, 
                                    alpha_path, mask_path])
        else:
            new_image_paths.append([image_path, matting_path])
    
    print('Write successfully to: {} and {}'.format(alpha_dir, mask_dir))
    print('Number of valid samples: ', len(new_image_paths))
    return new_image_paths


if __name__ == '__main__':
    root_dir = 'E://datasets/matting/Matting_Human_Half'
    train_txt_path = './train.txt'
    val_txt_path = './val.txt'
    image_paths = get_image_paths(root_dir=root_dir)
    image_paths = write_masks(image_paths, root_dir)
    train_image_paths, val_image_paths = split(image_paths)
    write_to_txt(train_image_paths, train_txt_path)
    write_to_txt(val_image_paths, val_txt_path)