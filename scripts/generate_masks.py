import os
import multiprocessing as mproc
import argparse
from tqdm import tqdm
import numpy as np
from endoanalysis.datasets import PointsDataset, parse_master_yaml
from endoanalysis.nucprop import NucleiPropagator
from endoanalysis.utils import generate_masks, check_masks_files_presence
from endoanalysis.utils import generate_masks_lists

parser = argparse.ArgumentParser(description="This script generates masks from images and keypoints using watershed algorithm")

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
    "--area_flags",
    dest = "area_flags",
    action="store_true",
    help="if provided, the area flags will be saved along with area masks",
)

parser.add_argument(
    "--new_master_dir",
    type=str,
    default="",
    help="if provided, new master yml and files lists will be saved in that dir. Cannot be used with --add_masters."
)

parser.add_argument(
    "--add_masters",
    dest = "add_masters",
    action="store_true",
    help="if provided, new master yml will be saved near initial master yml, and new files lists will be saved near orgignal labels lists. Cannot be used with --new_master_dir"
)

parser.add_argument(
    "--compress",
    dest = "compress",
    action="store_true",
    help="if provided, the resulting npz files will be compressed"
)

args = parser.parse_args()


MASTER_YAML = args.master
OVERWRITE = args.overwrite
NUM_WORKERS = args.workers
WINDOW_SIZE = args.window
MIN_AREA = args.min_area
MAX_AREA = args.max_area
AVERAGE_AREA = args.avg_area
AREA_FLAGS = args.area_flags
COMPRESS = args.compress
NEW_MASTER_DIR = args.new_master_dir
ADD_MASTERS = args.add_masters

if NEW_MASTER_DIR and ADD_MASTERS:
    raise Exception("NEW_MASTER_DIR and ADD_MASTERS cannot be used simultaneously")

if NEW_MASTER_DIR:    
    masks_dir = os.path.join(NEW_MASTER_DIR, "masks")
    os.makedirs(masks_dir, exist_ok=True)
else:
    masks_dir = None
    


lists = parse_master_yaml(MASTER_YAML)

cmap = {0: "red", 1: "green", 2: "blue"}

endo_dataset = PointsDataset(
    lists["images_lists"],
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
    check_masks_files_presence(endo_dataset, masks_dir)   
    

if NEW_MASTER_DIR:
    generate_masks_lists(lists, masks_dir, MASTER_YAML, NEW_MASTER_DIR)
elif ADD_MASTERS:
    generate_masks_lists(lists, masks_dir, MASTER_YAML, None)
    

print("Generating masks...")
if NUM_WORKERS == 1:
    with tqdm(total=len(endo_dataset)) as pbar:
        for image_i in range(len(endo_dataset)):
            generate_masks(image_i,endo_dataset, propagator, masks_dir, AREA_FLAGS, COMPRESS)
            pbar.update()
else:
    def wrapper_mproc(x):
        generate_masks(x, endo_dataset, propagator, masks_dir, AREA_FLAGS, COMPRESS)    
        
    with mproc.Pool(NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap(wrapper_mproc, range(len(endo_dataset))), total = len(endo_dataset)):
            pass
print("Done!")

if masks_dir:
    print("The masks are saved in: \n %s"%os.path.abspath(masks_dir))
else:
    print("The masks are saved near the labels files.")

    


