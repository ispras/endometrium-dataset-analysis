import argparse
import os
import numpy as np
from endoanalysis.datasets import MasksDataset
from endoanalysis.utils import parse_master_yaml, calculate_dab_values


parser = argparse.ArgumentParser(description="This script calculates dab values for a given dataset with masks.")

parser.add_argument(
    "--master", type=str,  help="yml file with paths to lists of images and masks", required=True
)

parser.add_argument(
    "--dab", type=str,  help="path to a file to store dab values. Should have .npy extention", required=True
)

args = parser.parse_args()
MASTER_PATH = args.master
DAB_PATH = args.dab

if not DAB_PATH.endswith(".npy"):
    raise Exception("Output dab file should have .npy extention, but the filename is\n %s"%DAB_PATH)

lists = parse_master_yaml(MASTER_PATH)
  
masks_dataset = MasksDataset(lists["image_lists"], lists["masks_lists"])
dabs_values = calculate_dab_values(masks_dataset)
np.save(DAB_PATH, dabs_values)