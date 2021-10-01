import os
import numpy as np

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
        return".".join(labels_path.split(".")[:-1] + ["npy"])
    else:
        return ".".join(os.path.basename(labels_path).split(".")[:-1] + ["npy"])

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
            raise Exception("File exists and overwrite mode is disabled: %s"%masks_path) 

def generate_masks(
    image_i, 
    endo_dataset,
    propagator,
    masks_dir
):
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
    """
    image = endo_dataset[image_i]["image"]
    keypoints = endo_dataset[image_i]["keypoints"]
    masks, borders, area_flags_image = propagator.generate_masks(image, keypoints, return_area_flags=True)
    masks = propagator.masks_to_image_size(image, masks, borders)
    labels_path = endo_dataset.labels_paths[image_i]
    masks_path = labels_path_to_mask_path(labels_path, masks_dir)
    
    with open(masks_path, "w+") as file:
        np.save(masks_path, masks)