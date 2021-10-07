import os
import argparse
from endoanalysis.datasets import parse_master_yaml, agregate_images_and_labels_paths
from endoanalysis.datasets import resize_dataset
import albumentations as A

class HWpair():
    def __init__(self, string):
        print(string)
        numbers = tuple(int(x) for x in string.split(','))
        if len(numbers)!=2:
            raise argparse.ArgumentError()
        self.h, self.w = numbers


    def tuple_form(self):
        return self.h, self.w



parser = argparse.ArgumentParser(description="This script resizes all images and keypoints in the dataset. Note, that the resize is performed inplace")

parser.add_argument(
    "--master", type=str,  help="yml file with paths to lists of images and keypoints", required=True
)

parser.add_argument(
    "--size", type=HWpair, help="the desired output size (h,w). Should be a pair comma separated integers, e.g. --size 256,256.", required=True
)

MASTER_YML_PATH = args.master
TARGET_SIZE = args.size.tuple_form()

lists = parse_master_yaml(MASTER_YML_PATH)
images_paths, keypoints_paths = agregate_images_and_labels_paths(lists["image_lists"], lists["labels_lists"])
resize_dataset(images_paths, keypoints_paths, target_size=TARGET_SIZE)