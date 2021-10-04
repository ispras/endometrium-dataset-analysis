import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def labels_path_to_mask_path(labels_path, masks_dir):
    """
    Converts path to file with labels to path to file with masks

    Paramterets
    -----------
    labels_path : str
        path to file with labels

    masks_dir : str
        path to dir to store masks in. If empty, the masks file will be created in the same dir as labels file.

    Returns
    -------
    masks_path : str
        path to file with masks
    """
    if not masks_dir:
        return ".".join(labels_path.split(".")[:-1] + ["npy"])
    else:

        masks_path = ".".join(os.path.basename(labels_path).split(".")[:-1] + ["npy"])
        return os.path.join(masks_dir, masks_path)


def check_masks_files_presence(endo_dataset, masks_dir):
    """
    Checks whether the masks files are already present at their destinations. If it founds any files, raises an exception.

    Parameters
    ----------
    endo_dataset : endoanalysis.datasets.PointsDataset
        dataset with the annotations

    masks_dir : str
        path to dir to store masks in. If empty, the masks file will be created in the same dir as labels file.
    """

    for image_i in range(len(endo_dataset)):
        labels_path = endo_dataset.labels_paths[image_i]
        masks_path = labels_path_to_mask_path(labels_path, masks_dir)
        if os.path.exists(masks_path):
            raise Exception(
                "File exists and overwrite mode is disabled: \n%s"
                % os.path.abspath(masks_path)
            )


def generate_masks(image_i, endo_dataset, propagator, masks_dir, area_flags=False):
    """
    Generates masks for all the annotated images in the dataset

    Parameters
    ----------
    image_i : int
        index of the image to generate mask for.

    endo_dataset : endoanalysis.datasets.PointsDataset
        dataset with the annotations

    masks_dir : str
        path to dir to store masks in. If empty, the masks file will be created in the same dir as labels file.

    area_flags : bool
        whether to store area flags. If True, the area flags will be stored with fnpy extention, which ca be read by the same means as npy files
    """
    image = endo_dataset[image_i]["image"]
    keypoints = endo_dataset[image_i]["keypoints"]
    masks, borders, area_flags_image = propagator.generate_masks(
        image, keypoints, return_area_flags=True
    )
    masks = propagator.masks_to_image_size(image, masks, borders)
    labels_path = endo_dataset.labels_paths[image_i]

    masks_path = labels_path_to_mask_path(labels_path, masks_dir)

    with open(masks_path, "wb+") as file:
        np.save(file, masks)

    if area_flags:
        area_flags_path = ".".join(masks_path.split(".")[:-1] + ["fnpy"])

        with open(area_flags_path, "wb+") as file:
            np.save(file, area_flags)


def load_masks_areas(paths_list):
    """
    Loads masks from the paths_list and caluclates their areas. kde

    Parameters
    ----------
    paths_list : list of str
        list of the relative paths to the masks npy files

    Returns
    -------
    masks_areas : ndarray
        areas of the masks
    """

    mask_areas = []
    #     print("Loading masks areas...")
    with tqdm(total=len(paths_list), desc="Loading mask areas") as pbar:
        for masks_path in paths_list:

            with open(masks_path, "rb") as file:
                masks = np.load(file)
                mask_areas += [x.sum() for x in masks]
                pbar.update()
    print("Done!")
    return np.array(mask_areas)


def decorate_areas_distr(fig, ax, impath="", caption="", dpi=300):
    """
    Puts some stuff on the figure and saves it if the path is provided
    """
    ax.set_xlabel("Area, pixels")
    ax.set_ylabel("Number of nuclei")
    ax.grid()
    if caption:
        ax.set_caption(caption)
    if impath:
        fig.savefig(impath, dpi=dpi)
