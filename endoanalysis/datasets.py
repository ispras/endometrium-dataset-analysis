import os
import cv2
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, cm
from endoanalysis.targets import Keypoints, keypoints_list_to_batch, load_keypoints
from endoanalysis.visualization import visualize_keypoints, visualize_masks


def extract_images_and_labels_paths(images_list_file, labels_list_file):
    """
    Extracts paths from the files with text lists of images and labels.

    Parameters
    ----------
    images_list_file: str
        path to the file with images ids
    labels_list_file: str
        path to the file with labels ids

    Returns
    -------
    images: list of str
        list of paths to images
    images: list of str
        list of paths to labels

    Note
    ----
    The lists inside the files should have the same length.
    Image id on i-th posidtion must coincide with labels id on i-th position.
    If these conditions are not met, the exception is raised.   
    """

    images_list_dir = os.path.dirname(images_list_file)
    labels_list_dir = os.path.dirname(labels_list_file)

    with open(images_list_file, "r") as images_file:
        images = images_file.readlines()
        images = [
            os.path.normpath(os.path.join(images_list_dir, x.strip())) for x in images
        ]
    with open(labels_list_file, "r") as labels_file:
        labels = labels_file.readlines()
        labels = [
            os.path.normpath(os.path.join(labels_list_dir, x.strip())) for x in labels
        ]

    check_images_and_labels_pathes(images, labels)

    return images, labels


def agregate_images_and_labels_paths(images_lists, labels_lists):
    """
    Aggregates images and labels paths from multiple files.

    Parameters
    ----------
    images_lists: str or list of str 
        if list, the list of paths to files with images ids. 
        If str, the single file with images ids.
    labels_lists: str or list of str
        if list, the list of paths to files with labels ids. 
        If str, the single file with labels ids.

    Returns
    -------
    images: list of str
        list of paths to images
    images: list of str
        list of paths to labels
    """

    if type(images_lists) != type(labels_lists):
        raise Exception(
            "images_list_files and labels_list_file should have the same type"
        )

    if type(images_lists) != list:
        images_lists = [images_lists]
        labels_lists = [labels_lists]

    images_paths = []
    labels_paths = []
    for images_list_path, labels_list_path in zip(images_lists, labels_lists):
        images_paths_current, labels_paths_current = extract_images_and_labels_paths(
            images_list_path, labels_list_path
        )
        images_paths += images_paths_current
        labels_paths += labels_paths_current

    return images_paths, labels_paths


def check_images_and_labels_pathes(images_paths, labels_paths):
    """
    Checks the consistency of images and labels paths.

    Parameters
    ----------
    images: list of str
        list of paths to images
    images: list of str
        list of paths to labels

    Note
    ----
    The lists checked should have the same length.
    Image id on i-th posidtion must coincide with labels id on i-th position.
    If these conditions are not met, the exception is raised.  
    """

    if len(images_paths) != len(labels_paths):
        raise Exception("Numbers of images and labels are not equal")

    for image_path, labels_path in zip(images_paths, labels_paths):
        filename_image = os.path.basename(image_path)
        filename_labels = os.path.basename(labels_path)

        if ".".join(filename_image.split(".")[:-1]) != ".".join(
            filename_labels.split(".")[:-1]
        ):
            raise Exception(
                "Different dirnames found: \n %s\n  %s" % (images_paths, labels_paths)
            )


def load_image(image_path):
    """
    Loads image from a given path.

    Parameters
    ----------
    image_path: str
        path to image to load
    
    Returns
    -------
    image: ndarray
        the loaded image in RGB mode, the shape is (H, W, C)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class PointsDataset:

    """
    Dataset providing access to images and keypoints.
    Made in pytorch dataset style.

    Parameters
    ----------
    images_lists: str or list of str 
        if list, the list of paths to files with images ids. 
        If str, the single file with images ids.
    labels_lists: str or list of str
        if list, the list of paths to files with labels ids. 
        If str, the single file with labels ids.
    class_colors: dict of tuple
        colors for classes in visualisation
    """
    def __init__(
        self,
        images_list,
        labels_list
    ):

        self.images_paths, self.labels_paths = agregate_images_and_labels_paths(
            images_list,
            labels_list,
        )
        self.class_colors = class_colors
        self._keypoints_dtype = np.float

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, x):
        """
        Provides dataset item.

        Parameters
        ----------
        x: int
            sample id
        
        Returns
        -------
        sample: dict
            dictionary with two keys: "image" (ndarray) and "keypoints" (endoanalysis.targets.Keypoints)
        """
        image = load_image(self.images_paths[x])
        keypoints = load_keypoints(self.labels_paths[x])

        class_labels = [x[-1] for x in keypoints]
        keypoints_no_class = [x[:-1] for x in keypoints]

        keypoints = [
            np.array(y + (x,)) for x, y in zip(class_labels, keypoints_no_class)
        ]

        if keypoints:
            keypoints = np.stack(keypoints)
        else:
            keypoints = np.empty((0, 3))

        sample = {
            "keypoints": Keypoints(keypoints.astype(self._keypoints_dtype)),
            "image": image
            }

        return sample

    def visualize(
        self,
        x,
        show_labels=True,
        labels_kwargs={"radius": 3, "alpha": 1.0, "ec": (0, 0, 0)},
        class_colors={x: cm.Set1(x) for x in range(10)},
    ):
        """
        Visulalizes sample with given id. 
        The data visualised is the ones which are got with self.__getitem__ method.

        Parameters
        ----------
        x: int
            sample id
        show_labels: bool
            whether to show keypoints
        labels_kwargs: dict
            dictionary of parameters for keypoints
        class_colors: dict of tuple
            RGB colors for different classes

        See also
        --------
        endoalaysis.visualisation.visualize_keypoints
        """

        sample = self[x]
        image = sample["image"]

        if show_labels:
            keypoints = sample["keypoints"]
        else:
            keypoints = Keypoints(np.empty((0, 3)))

        _ = visualize_keypoints(
            image,
            keypoints,
            class_colors=class_colors,
            circles_kwargs=labels_kwargs,
        )

    def collate(self, samples):
        """
        Composes batch from multiple samples

        Parameters
        ----------
        samples: list
            list of samples provided by self.__getitem__ method
        
        Returns
        -------
        batch: dict
            dictionary with two keys: "image" (ndarray) and "keypoints" (endoanalysis.targets.KeypointsBatch)
        
        """
        images = [x["image"] for x in samples]
        keypoints_groups = [x["keypoints"] for x in samples]

        batch = {
            "image": np.stack(images, 0),
            "keypoints": keypoints_list_to_batch(keypoints_groups),
        }

        return batch


class MasksDataset:
    def __init__(
        self, images_list, masks_list
    ):
        """
        Dataset providing access to images and keypoints.
        Made in pytorch dataset style.
    
    Parameters
    ----------
    images_lists: str or list of str 
        if list, the list of paths to files with images ids. 
        If str, the single file with images ids.
    masks_lists: str or list of str
        if list, the list of paths to files with masks ids. 
        If str, the single file with masks ids.
        """

        if type(images_list) != type(masks_list):
            raise Exception("images_list and masks_list should have the same type")

        if type(images_list) != list:
            images_list = [images_list]
            masks_list = [masks_list]

        self.images_paths = []
        self.masks_paths = []
        for images_list_path, masks_list_path in zip(images_list, masks_list):
            images_paths_current, masks_paths_current = extract_images_and_labels_paths(
                images_list_path, masks_list_path
            )
            self.images_paths += images_paths_current
            self.masks_paths += masks_paths_current
        self.cmap_kwargs = cmap_kwargs

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, x):
        """
        Provides dataset item.

        Parameters
        ----------
        x: int
            sample id
        
        Returns
        -------
        sample: dict
            dictionary with two keys: "image" (ndarray) and "masks" (ndarray)
        """
        masks_array = np.load(self.masks_paths[x])

        return {
            "image": load_image(self.images_paths[x]),
            "masks": masks_array["masks"],
            "classes": masks_array["classes"],
        }

    def visualize(self, x, cmap_kwargs={"cmap": cm.Set1, "period": 8}):
        """Visualises a sample with a given id"""
        sample = self[x]
        visualize_masks(sample["image"], sample["masks"])


def resize_dataset(image_paths, labels_paths, target_size=(256, 256)):
    """
    Resizes dataset to a given image size. The resize is made inplace.

    Parameters
    ----------
    image_paths: list os str
        paths to images
    labels_paths: list of str
        paths to labels, should have the same length as labels_paths.
        Assumed to be unique
    target_size: tuple of int
        output image size, the format is (x_size, y_size)
    """

    image_processed = {x: False for x in image_paths}
    image_sizes = {}

    if len(labels_paths) != len(np.unique(labels_paths)):
        raise Exception("There are repetaing labels paths.")

    with tqdm(total=len(image_paths)) as pbar:
        for image_path, labels_path in zip(image_paths, labels_paths):

            if not image_processed[image_path]:
                image = cv2.imread(image_path)
                image_h, image_w, _ = image.shape
                image_new = cv2.resize(
                    image, target_size, interpolation=cv2.INTER_LINEAR
                )
                cv2.imwrite(image_path, image_new)
                image_processed[image_path] = True
                image_sizes[image_path] = image.shape[0:2]
            else:
                image_h, image_w = image_sizes[image_path]

            keypoints = load_keypoints(labels_path)
            keypoints = np.array(keypoints).astype(int)

            if len(keypoints):

                keypoints[:, 0][keypoints[:, 0] == image_w] = image_w - 1
                keypoints[:, 1][keypoints[:, 1] == image_h] = image_h - 1
                x_coords_new = keypoints[:, 0] * target_size[0] / image_w
                y_coords_new = keypoints[:, 1] * target_size[1] / image_h
                x_coords_new = x_coords_new.astype(int)
                y_coords_new = y_coords_new.astype(int)

                keypoints_new = np.vstack(
                    [x_coords_new, y_coords_new, keypoints[:, 2]]
                ).T

            else:
                keypoints_new = []

            labels_lines = [
                " ".join([str(x), str(y), str(class_id)]) + " \n"
                for x, y, class_id in keypoints_new
            ]

            os.remove(labels_path)

            with open(labels_path, "w+") as labels_file:
                labels_file.writelines(labels_lines)

            pbar.update()


def parse_master_yaml(yaml_path):
    """
    Imports master yaml and converts paths to make the usable from inside the script

    Parameters
    ----------
    yaml_path : str
        path to master yaml from the script

    Returns
    -------
    lists : dict of list of str
        dict with lists pf converted paths
    """
    with open(yaml_path, "r") as file:
        lists = yaml.safe_load(file)

    for list_type, paths_list in lists.items():
        new_paths_list = []
        for path in paths_list:
            new_path = os.path.join(os.path.dirname(yaml_path), path)
            new_path = os.path.normpath(new_path)
            new_paths_list.append(new_path)
        lists[list_type] = new_paths_list

    return lists
