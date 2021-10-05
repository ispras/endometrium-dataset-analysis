import os
import multiprocessing as mproc
import argparse
from tqdm import tqdm
import yaml
import numpy as np
from endoanalysis.datasets import PointsDataset
from endoanalysis.nucprop import NucleiPropagator
from endoanalysis.utils import generate_masks, check_masks_files_presence
from endoanalysis.utils import generate_masks_lists

parser = argparse.ArgumentParser(description="This script generates masks from images and keypoints using watershed algorythm")

parser.add_argument(
    "--master", type=str,  help="yml file with paths to lists of images and labels", required=True
)
parser.add_argument(
    "--overwrite", dest='overwrite', action='store_true', help="whether overwrite masks or not"
)
parser.add_argument(
    "--workers", type=int, default=1, help="number of workers to work in parallel"
)
parser.add_argument(
    "--window",
    type=int,
    default=100,
    help="size of the window in pixels for NucleiPropagator",
)
parser.add_argument(
    "--avg_area",
    type=int,
    default=10,
    help="average area of the mask. If mask is too big or too small, the circle of this area will be created",
)
parser.add_argument(
    "--min_area",
    type=int,
    default=0,
    help="minimal accepted watershed mask area. If watershed mask is smaller, a circile of average area will be created",
)
parser.add_argument(
    "--max_area",
    type=int,
    default=np.inf,
    help="maximal accepted watershed mask area. If watershed mask is larger, a circile of average area will be created",
)
parser.add_argument(
    "--masks_dir",
    type=str,
    default="",
    help="if provided, all the masks will be stored there. If not provided, masks files will be stored near the labels files",
)
parser.add_argument(
    "--area_flags",
    dest = "area_flags",
    action="store_true",
    help="if provided, the area flas will be saved along with area masks",
)
parser.add_argument(
    "--masks_lists",
    dest = "masks_lists",
    action="store_true",
    help="if provided, the txt files with masks lists will be created near the files with labels lists. Also new master yml will be created neare the original master yml",
)

parser.add_argument(
    "--compress",
    dest = "compress",
    action="store_true",
    help="if provided, the resulting npz files will be compressed",
)

args = parser.parse_args()


MASTER_YAML = args.master
OVERWRITE = args.overwrite
NUM_WORKERS = args.workers
WINDOW_SIZE = args.window
MIN_AREA = args.min_area
MAX_AREA = args.max_area
AVERAGE_AREA = args.avg_area
MASKS_DIR = args.masks_dir
AREA_FLAGS = args.area_flags
MASKS_LISTS = args.masks_lists
COMPRESS = args.compress

if not os.path.exists(MASKS_DIR) and MASKS_DIR:
    os.makedirs(MASKS_DIR, exist_ok=True)

with open(MASTER_YAML, "r") as file:
    lists = yaml.safe_load(file)

cmap = {0: "red", 1: "green", 2: "blue"}

endo_dataset = PointsDataset(
    lists["image_lists"],
    lists["labels_lists"],
    keypoints_dtype=float,
    cmap_kwargs={"cmap": cmap},
)

propagator = NucleiPropagator(
    window_size=WINDOW_SIZE,
    min_area=MIN_AREA,
    max_area=MAX_AREA,
    average_area=AVERAGE_AREA,
)


if not OVERWRITE:
    check_masks_files_presence(endo_dataset, MASKS_DIR)   
    
if MASKS_LISTS:
    generate_masks_lists(lists, MASKS_DIR, MASTER_YAML)


print("Generating masks...")
if NUM_WORKERS == 1:
    with tqdm(total=len(endo_dataset)) as pbar:
        for image_i in range(len(endo_dataset)):
            generate_masks(image_i,endo_dataset, propagator, MASKS_DIR, AREA_FLAGS, COMPRESS)
            pbar.update()
else:
    def wrapper_mproc(x):
        generate_masks(x, endo_dataset, propagator, MASKS_DIR, AREA_FLAGS, COMPRESS)    
        
    with mproc.Pool(NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap(wrapper_mproc, range(len(endo_dataset))), total = len(endo_dataset)):
            pass
print("Done!")

if MASKS_DIR:
    print("The masks are saved in: \n %s"%os.path.abspath(MASKS_DIR))
else:
    print("The masks are saved near the labels files.")

    


