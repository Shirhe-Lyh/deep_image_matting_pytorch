# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:22:00 2019

@author: shirhe-lyh
"""

import os


def get_image_paths(root_dir, dataset='Training_set'):
    image_paths_dict = {}
    matting_paths_dict = {}
    sub_root_dir = os.path.join(root_dir, dataset)
    for root, dirs, files in os.walk(sub_root_dir):
        if not files:
            continue
        
        for file in files:
            file_path = os.path.join(root, file)
            file_path = file_path.replace('\\', '/')
            file_name = file.split('.')[0]
            dir_name = file_path.split('/')[-2]
            if dir_name.startswith('fg'):
                image_paths_dict[file_name] = file_path
            if dir_name.startswith('alpha'):
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


def write_to_txt(image_paths, txt_path, delimiter='@'):
    if image_paths is None:
        return
    
    with open(txt_path, 'w') as writer:
        for element in image_paths:
            line = delimiter.join(element)
            writer.write(line + '\n')
    print('Write successfully to: ', txt_path)