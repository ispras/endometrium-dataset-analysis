import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, cm
import albumentations as A

from endoanalysis.keypoints import KeypointsTruthArray, KeypointsTruthBatchArray, keypoints_list_to_batch, load_keypoints


def label_path_from_image_path(image_path):

    image_path_parts = os.path.normpath(image_path).split(os.sep)
    label_path_parts = ["labels"] + image_path_parts[1:]

    label_path = os.path.join(*label_path_parts)
    label_path = label_path.split(".")[:-1]
    label_path = ".".join(label_path + ["txt"])
    return label_path

    
def parse_files_list(path_to_file_with_list):
    images_paths = []
    labels_paths = []
    dataset_dir = os.path.dirname(path_to_file_with_list)

    with open(path_to_file_with_list, "r") as image_list_file:
        for line in image_list_file:

            image_path = line.strip()
            if not (
                image_path.startswith("./images") or image_path.startswith("images")
            ):
                raise Exception("Wrong file path spectification: %s" % image_path)

            if image_path.startswith("./images"):
                image_path = image_path[2:]

            images_paths.append(image_path)
            
            
    images_paths.sort()
    labels_paths = [os.path.join(dataset_dir, label_path_from_image_path(x)) for x in images_paths]
    images_paths = [os.path.join(dataset_dir, x) for x in images_paths]

    return images_paths, labels_paths


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class PointsDataset:
    def __init__(
        self,
        path_to_file_with_list,
        keypoints_dtype=np.int16,
    ):

        self.path_to_file_with_list = path_to_file_with_list
        self.keypoints_dtype = keypoints_dtype
        # reading images_list
        self.images_paths, self.labels_paths = parse_files_list(path_to_file_with_list)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, x):
        
       
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
            keypoints = np.empty((0,3))

        to_return = {
            "keypoints": KeypointsTruthArray(keypoints.astype(self.keypoints_dtype))
        }


        to_return["image"] = image

        return to_return


    def visualize(
        self,
        x,
        show_labels=True,
        image_key="image",
        print_labels=False,
        labels_kwargs={"radius": 3, "alpha": 1.0},
        colormap_kwargs = {"cmap": cm.Set1, "period": 8} 
    ):

        sample = self[x]
        image = sample[image_key]
        keypoints = sample["keypoints"]
        fig, ax = plt.subplots()

        ax.imshow(image)
        x_coords = keypoints.x_coords()
        y_coords = keypoints.y_coords()
        classes = keypoints.classes()

        if show_labels:
            for center_x, center_y, obj_class in zip(x_coords, y_coords, classes):
                
                if colormap_kwargs["period"] is None:
                    color = colormap_kwargs["cmap"](obj_class)
                else:
                    color = colormap_kwargs["cmap"](obj_class % colormap_kwargs["period"])
                
                patch = patches.Circle((center_x, center_y), color=color, **labels_kwargs)

                ax.add_patch(patch)
        if print_labels:
            for keypoint_i, keypoint in enumerate(keypoints_tuples):
                print("%i. %i, %f, %f, %f, %f" % ((keypoint_i,) + keypoint))
                
                
    def collate(self):

        images = [self[x]["image"] for x in range(len(self))]
        keypoints_groups = [self[x]["keypoints"] for x in range(len(self))]

        return_dict = {
            "image": np.stack(images, 0),
            "keypoints": keypoints_list_to_batch(keypoints_groups),
        }

        return return_dict

                
                
def resize_dataset(dataset_main_file_path, target_size=(256,256)):
    
    transorm = A.Compose([A.Resize(height=target_size[0], width=target_size[1])], keypoint_params=A.KeypointParams(format="xy"))
    dataset_dir = os.path.dirname(dataset_main_file_path)
    
    with open(dataset_main_file_path, "r") as main_file:
        lines = main_file.readlines()
        for line in lines:
            image_path = os.path.join(dataset_dir, line.strip())

            image = image = cv2.imread(image_path)
            labels_path  = label_path_from_image_path(line)
            labels_path = os.path.join(dataset_dir, labels_path)
       
            keypoints = load_keypoints(labels_path)

            if keypoints:
                
                keypoints = np.array(keypoints)
                coords = keypoints[:,0:2]
                classes = keypoints[:,2]
            else: 
                coords = []
            transformed = transorm(image = image, keypoints=coords)

            cv2.imwrite(image_path, transformed["image"])
            
            labels_lines = [
                " ".join([str(int(y)) for y in label] + [str(class_id)]) + " \n"
                for label, class_id in zip(transformed["keypoints"], classes)
                ]
            
            os.remove(labels_path)

            with open(labels_path, "w+") as labels_file:
                labels_file.writelines(labels_lines)
                
    