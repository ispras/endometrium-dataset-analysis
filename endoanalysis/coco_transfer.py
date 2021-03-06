import os
import json
import shutil
from posixpath import normpath
import cv2
from endoanalysis.datasets import parse_master_yaml


def make_coco(path_to_yaml, output_dir=None, anno_name='annotation', name_image_folder='images', return_ano=False):
    """ Remake endometrium dataset to coco format

    Args:
        path_to_yaml (str): The abs or rel path to the yaml file with lists of images and labels annotations.
        output_dir (bool): The dir in which dataset will be copied.
        anno_name (str): The name of json annotation.
        name_image_folder (str): The name of folder with images in coco dataset.
        return_ano (bool): The flag of returning dict with annotation.

    Returns:
        dict {str: list, str: list} or None: The dict with annotation can be returned,
            but default option is returning None.
    """
    coco_ano = {}
    lists = parse_master_yaml(path_to_yaml)
    images_list = lists['images_lists']
    labels_list = lists['labels_lists']
    images = []
    annotation = []
    image_check_list = set()
    labels_check_list = set()
    
    for images_path in images_list:
        dir_path = os.path.abspath(os.path.split(images_path)[0])
        with open(images_path, 'r') as images_list_again:
            for image_path in images_list_again:
                image = {}
                image_name = image_path.rstrip()
                norm_path = os.path.normpath(os.path.join(dir_path, image_name))
                img = cv2.imread(norm_path)
                image['height'] = img.shape[0]
                image['width'] = img.shape[1]
                image['file_name'] = os.path.split(image_name)[1]
                image['id'] = int(os.path.splitext(image['file_name'])[0])
                if image['id'] not in image_check_list:
                    images.append(image)
                    if output_dir:
                        shutil.copy(norm_path, os.path.join(output_dir, name_image_folder, image['file_name']))
                    image_check_list.add(image['id'])

    for labels_path in labels_list:
        dir_path = os.path.abspath(os.path.split(labels_path)[0])
        with open(labels_path, 'r') as labels_list_again:
            for label_path in labels_list_again:
                label_name = label_path.rstrip()
                norm_path = os.path.normpath(os.path.join(dir_path, label_name))
                image_id = int(os.path.splitext(os.path.split(label_name)[1])[0])
                if image_id in image_check_list and image_id not in labels_check_list:
                    labels_check_list.add(image_id)
                    with open(norm_path, 'r') as labels_for_image:
                        for label in labels_for_image:
                            labels = {}
                            labels['num_keypoints'] = 1
                            keypoints = label.strip().split(" ")
                            keypoints = list(int(float(x)) for x in keypoints)
                            labels['keypoints'] = keypoints[:-1]
                            labels['image_id'] = image_id
                            labels['bbox'] = labels['keypoints'] + labels['keypoints']
                            labels['category_id'] = keypoints[-1]
                            annotation.append(labels)
        
    coco_ano['images'] = images
    coco_ano['annotations'] = annotation
    if output_dir:
        with open(f'{output_dir}/{anno_name}.json', 'w') as anno_file:
            json.dump(coco_ano, anno_file)
    if return_ano:
        return coco_ano