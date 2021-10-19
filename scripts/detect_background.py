import numpy as np
import os
import cv2
import sys
import argparse

def is_background(image):
    """Whether given BGR image is mostly white"""
    gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (np.mean(gs) >= 205 and np.std(gs) <= 41) or np.count_nonzero(gs >= 205) / gs.size >= 0.9

def is_background_gray(gs):
    """Whether given Grayscale image is mostly white"""
    return (np.mean(gs) >= 220 and np.std(gs) <= 35) or np.count_nonzero(gs >= 220) / gs.size >= 0.95

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This script decides, whether images from given list are mostly background." + 
        " Background: 1, informative image: 0"
    )
    parser.add_argument('file_list', help='File with list of images to check')
    args = parser.parse_args()

with open(args.file_list, "r") as file:
    for image_path in file.readlines():

        image_path = os.path.join(os.path.dirname(args.file_list), image_path.strip())
        image_path = os.path.normpath(image_path)
        image = cv2.imread(image_path)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = 0
        if is_background(image):
            result += 0.85
        if is_background_gray(image_gs):
            result += 0.15
        result = round(result)
        print(result)