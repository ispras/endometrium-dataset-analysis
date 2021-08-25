import numpy as np


class KeypointsArray(np.ndarray):
    """
    Subclass of ndarray designed to store the keypoints.
    This is the base class for  keypoints arrays of different types.

    Parameters
    ----------
    input_array : ndarray
        the array to create KeypointsArray from.
        Must have float dtype and the shape (num_keypoints, expected_param_number).
        If there is no_keypoints, it should have the shape (0,).


    Methods
    -------
    conf_striped()
        returns the keypoints without the confidences.
    confidences()
        returns the keypoints confidences
    x_coords()
        returns the x coordinates of the keypoints as ints.
    y_coords()
        returns the x coordinates of the keypoints as ints.
    classes()
        returns the keypoints class labels
    specs()
        returns the strig describing keypoints format.

    Note
    ----
    The intialisations procedures (like checks of array shape and type) are done
    not in the  __array_filnalise__ as recommended in

    https://numpy.org/doc/stable/user/basics.subclassing.html

    but in the __init__ method. This is done to avoid unneccessary checks,
    for exmaple when slicing keypoints.
    """

    def __new__(cls, input_array):

        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, *args, **kwargs):
        self.expected_param_number = 3
        self._check_shape()
        self._check_dtype()

#     def _check_shape(self):
#         if len(self.shape) == 1 and self.shape[0] != 0:
#             raise Exception("If KeypointsArray is empty, it should be one dimensional")
#         elif len(self.shape) == 2 and self.shape[1] != self.expected_param_number:
#             raise Exception(
#                 "Non-empty %s should have the shape (num_images, %i)"
#                 % (type(self), self.expected_param_number)
#             )
    def _check_shape(self):
#         print(self.shape)
        if len(self.shape) != 2:
            raise Exception(
                " %s should have the shape (num_images, %i)"
                % (type(self), self.expected_param_number)
            )
        elif self.shape[1] != self.expected_param_number and self.shape[1] != 0:

            raise Exception(
                "The shape is %s, but the expected number of paramteres is %i"
                % (self.shape, self.expected_param_number)
            )

    def _check_dtype(self):
        if self.dtype != float:
            raise Exception("KeypointsArray should have dtype float")

    def conf_striped(self):
        """
        Returns the keypoints without the confidence.

        Returns
        -------
        keypoints_no_conf : ndarray of float
            2D array with confidences of shape (num_keypoints, 3).
            3 stands for (x, y, class).
        """
        raise NotImplementedError()

    def confidences(self):
        """
        Returns the keypoints confidences.

        Returns
        -------
        confidences : ndarray of float
            1D array with confidences.
        """
        raise NotImplementedError()

    def x_coords(self):
        """
        Returns the x coordinates of the keypoints.

        Returns
        -------
        confidences : ndarray of int
            1D array with x coordinates.
        """
        raise NotImplementedError()

    def y_coords(self):
        """
        Returns the y coordinates of the keypoints

        Returns
        -------
        confidences : ndarray of int
            1D array with x coordinates.
        """
        raise NotImplementedError()

    def classes(self):
        """
        Returns the class labels of the keypoints .

        Returns
        -------
        confidences : ndarray of int
            1D array with class labels.
        """
        raise NotImplementedError()

    def specs(self):
        """
        Returns the specifications of the keypoints array.

        Returnshttps://numpy.org/doc/stable/user/basics.subclassing.html
        -------

        specs : str
            keypoints array format specifications.
        """
        raise NotImplementedError()


class KeypointsTruthArray(KeypointsArray):
    """
    Keypoints array for storing the ground truth keypoints

    Note
    ----
    Ground truth keypoints have no confidence data, so the shape should be (num_keypoints, 3) where 3 stands for  (x, y, class).
    If the confidences() is called, the 1d ndarray of 1. with the len of num_keypoints will be returned.

    See also
    --------
    pointdet.utils.keypoints.KeypointsArray
        the base class with api specifications
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def conf_striped(self):
        return self

    def confidences(self):
        return np.ones(len(self))

    def x_coords(self):
        return self[:, 0].astype(int)

    def y_coords(self):
        return self[:, 1].astype(int)

    def classes(self):
        return self[:, 2].astype(int)

    def specs(self):
        return "(x, y, class)"


class KeypointsBatchArray(KeypointsArray):
    """
    Base class for keypointsfor image batch.

    Methods
    -------
    image_labels()
        returns labels of images in th batch for each keypoint
    from_image(image_i)
        returns the keypoints from a specific image

    Note
    ----
    The first number for each keypoit must indicate image label, so the format should be like
    (image_label, x, y, class) or (image_label, x, y, class, confidence)

    See also
    --------
    pointdet.utils.keypoints.KeypointsArray
        the base class with api specifications
    """

    def image_labels(self):
        """
        Returns labels of images in th batch for each keypoint

        Returns
        -------
        image_labels : ndarray of int
            image labels for all keypoints
        """
        raise NotImplementedError()

    def _prepare_array_from_image(self, image_i):
        """
        Prepares the array from a given image

        Parameters
        ----------
        image_i : int
            label of the image to take the keypoints from is. If not present, the excpetion will be raised.

        Returns
        -------
        keypoints : ndarray
            image labels for all keypoints.
        """
        if self.shape[0] == 0:
            return self[:,1:]
        mask = self.image_labels() == image_i

        if mask.sum() == 0:
            return np.empty((0, self.shape[1] - 1))
#             raise Exception("No image with label %i" % int(image_i))
        return np.array(np.array(self[mask][:, 1:]))

    def from_image(self, image_i):
        """
        Returns the keypoints from a specific image.

        Parameters
        ----------
        image_i : int
            label of the image to take the keypoints from is. If not present, the excpetion will be raised.

        Returns
        -------
        keypoints : KeypointsTruthArray or KeypointsPredArray
            image labels for all keypoints.
        """

        raise NotImplementedError()

    def image_labels(self):
        '''
        Returns image_labels
        '''
        return self[:, 0].astype(int)
    
    def num_images(self):
        '''
        Returns number of images in a batch
        '''

        return len(np.unique(self.image_labels()))
    

class KeypointsTruthBatchArray(KeypointsBatchArray):
    """
    Keypoints array for storing the predicted keypoints for image bach

    Note
    ----
    The shape should be (num_keypoints, 4) where 5 stands for  (image_label, x, y, class)

    See also
    --------
    pointdet.utils.keypoints.KeypointsArray
        the base class with api specifications
    pointdet.utils.keypoints.KeypointsBatchArray
        the base class with api specifications
    """

    def __init__(self, *args, **kwargs):
        self.expected_param_number = 4
        self._check_shape()
        self._check_dtype()

    def conf_striped(self):
        return self[:, 0:3].astype(int)

    def confidences(self):
        return np.ones(len(self))

    def x_coords(self):
        return self[:, 1].astype(int)

    def y_coords(self):
        return self[:, 2].astype(int)

    def classes(self):
        return self[:, 3].astype(int)

    def specs(self):
        return "(image_i, x, y, class)"

    def from_image(self, image_i):
        return KeypointsTruthArray(self._prepare_array_from_image(image_i))


def load_keypoints(file_path):
    """
    Load keypoints from a specific file as tuples

    Parameters
    ----------
    file_path : str
        path to the file with keypoints

    Returns
    -------
    keypoints : list of tuples
        list of keypoint tuples in format (x, y, obj_class)

    Note
    ----
    This function serves as helper for the pointdet.utils.dataset.PointsDataset class
    and probably should be moved there
    """

    keypoints = []

    with open(file_path, "r") as labels_file:
        for line in labels_file:
            line_contents = line.strip().split(" ")
            line_floated = tuple(int(x) for x in line_contents)
            x_center, y_center, obj_class = tuple(line_floated)
            keypoint = x_center, y_center, obj_class
            keypoints.append(keypoint)

    return keypoints

def keypoints_list_to_batch(keypoints_list):
    '''
    Transforms list of keypoints to KeypointsBatch object
    
    Parameters
    ----------
    keypoints_list : list of KeypointsTruthArray or KeypointsPredArray
        keypoints list to transform
    
    Returns
    -------
    batch : KeypointsTruthBatchArray or KeypointsPredBatchArray
        transformed batch
    '''
    keypoints_return = []
    
    
    current_type = type(keypoints_list[0])

    if current_type == KeypointsTruthArray:
        batch_type = KeypointsTruthBatchArray
    elif current_type == KeypointsPredArray:
        batch_type = KeypointsPredBatchArray
    else:
        raise Exception("Unsupported keypoints type %s"%current_type)
        
    for image_i, keypoints in enumerate(keypoints_list):
        if len(keypoints) != 0:
            image_labels = (np.ones(keypoints.shape[0]) * image_i)[:,np.newaxis]
            keypoints = np.hstack([image_labels, keypoints])
            keypoints_return.append(keypoints)

    return batch_type(np.concatenate(keypoints_return, 0))
    