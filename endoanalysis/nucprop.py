import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.filters import threshold_multiotsu

from endoanalysis.targets import Keypoints


def check_bounds(coord, bounds):
    """
    Check whether the coords inside the give boundaries.

    Parameters
    ----------
    coord: int or float
        coord to check.
    bounds: tuple of int or float
        boundaries within wich the coord should lay.
        bounds[1] should be greatere than bounds[0].

    Returns
    -------
    result: bool
        wheather the coord inside the boundaries.
    """

    if bounds[0] > bounds[1]:
        raise Exception("Unordered bounds")
    if bounds[0] < coord < bounds[1]:
        return True
    else:
        return False


def make_window(image, image_keypoints, center_keypoint_i, window_size):
    """
    Clips image around given keypoint and returns all the keypoints which lay within clipped window.

    Parameters
    ----------
    image: ndarray
        the initial image
    image_keypoints: Keypoints
        all keypoints corresponding to the image.
    center_keypoint_i: int
        index of the keypoint to make window around.
    window_size: int
        Desired size of the window. Note, that if central point is close to the border,
        the window size will be reduced.

    Returns
    -------
    image_clipped: ndarray
        fragment of the image around central point.

    keypoints_clipped: Keypoints
        keypoints within  the clipped area. Central keypoint is always under the 0th index

    borders: tuple of int
        4 integers refering to a window borders: bound_x_left, bound_x_right, bound_y_top, bound_y_bottom
    """
    window_half_size = np.floor(window_size / 2).astype(int)
    center_keypoint = image_keypoints[center_keypoint_i]

    x_coords = image_keypoints.x_coords().astype(int)
    y_coords = image_keypoints.y_coords().astype(int)
    classes = image_keypoints.classes().astype(int)

    bound_x_left = max(x_coords[center_keypoint_i] - window_half_size, 0)
    bound_x_right = min(x_coords[center_keypoint_i] + window_half_size, image.shape[1])

    bound_y_top = max(y_coords[center_keypoint_i] - window_half_size, 0)
    bound_y_bottom = min(y_coords[center_keypoint_i] + window_half_size, image.shape[0])

    image_clipped = image[bound_y_top:bound_y_bottom, bound_x_left:bound_x_right, :]

    new_keypoint = [
        min(x_coords[center_keypoint_i] - bound_x_left, image_clipped.shape[1] - 1),
        min(y_coords[center_keypoint_i] - bound_y_top, image_clipped.shape[0] - 1),
        classes[center_keypoint_i],
    ]

    keypoints_clipped = [new_keypoint]

    remaining_keypoints = set(range(len(image_keypoints))) - set([center_keypoint_i])

    for keypoint_i in remaining_keypoints:
        if check_bounds(
            x_coords[keypoint_i], (bound_x_left, bound_x_right)
        ) and check_bounds(y_coords[keypoint_i], (bound_y_top, bound_y_bottom)):
            new_keypoint = image_keypoints[keypoint_i]
            new_keypoint = [
                x_coords[keypoint_i] - bound_x_left,
                y_coords[keypoint_i] - bound_y_top,
                classes[keypoint_i],
            ]
            keypoints_clipped.append(new_keypoint)

    keypoints_clipped = Keypoints(np.array(keypoints_clipped, dtype=float))
    return (
        image_clipped,
        keypoints_clipped,
        (bound_x_left, bound_x_right, bound_y_top, bound_y_bottom),
    )


def compute_image_grad(image):
    """
    Compute image gradient. Each pixel stores the maximum from x- and y- gradients.

    Parameters
    ----------
    image: nd.array
        image to take the gradient from. If it has multiple channales,
        the gradyscale version is taken before computeng the gradient.

    Returns
    -------
    image_grad_abs: ndarray
        image gradient (maxim values from x- and x-y gradients at a given point).

    """
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    else:
        image_gray = image.astype(np.uint8)

    image_grad_x = np.abs(cv2.Sobel(image_gray, cv2.CV_64F, 1, 0))
    image_grad_y = np.abs(cv2.Sobel(image_gray, cv2.CV_64F, 0, 1))
    image_grad_x = image_grad_x * 255 / image_grad_x.max()
    image_grad_y = image_grad_y * 255 / image_grad_y.max()
    image_grad_abs = np.where(
        image_grad_x > image_grad_y, image_grad_x, image_grad_y
    ).astype(np.uint8)
    return image_grad_abs


def get_adjacent_points(image, x, y, size=1):
    """
    Get the coordinates of the squire neighborhood of image from a given point.

    Parameters
    ----------
    image: ndarray
        image containing the initial point. Needed mostly for the shap data.
    x: int
        x coordinate of the target point.
    y: int
        y coordinate of the target point.
    size: int
        the disired size of the neighborhood.

    Returns
    -------
    x_coors: ndarray
        x coordinates of the neighborhood points.
    y_coors: ndarray
        y coordinates of the neighborhood points.

    Note
    ----
    If the target point is close to the boundaries, the neighborhood will be clipped.
    """

    dx_min = -min(size, x)
    dy_min = -min(size, y)
    dx_max = min(image.shape[1] - x - 1, size)
    dy_max = min(image.shape[0] - y - 1, size)
    x_coords = []
    y_coords = []
    for dx, dy in itertools.product(range(dx_min, dx_max), range(dy_min, dy_max)):
        x_coords.append(x + dx)
        y_coords.append(y + dy)

    return x_coords, y_coords


def watershed_mask(image, keypoints, num_otsu_classes=3):
    """
    Generate watershed mask on the image using keypoints as cavities centers.

    Parameters
    ----------
    image: ndarray
        image to generate the whatershed mask on.
    keypoints : Keypoints
        keypoints marking the centers of the cavities.

    Returns
    -------
    mask: ndarray of int
        whatershed mask.

    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    markers = np.zeros_like(image_gray)
    markers[
        image_gray
        > threshold_multiotsu(image_gray, classes=num_otsu_classes)[
            num_otsu_classes - 2
        ]
    ] = 1

    x_coords = keypoints.x_coords()
    y_coords = keypoints.y_coords()
    for keypoint_i, (x_c, y_c) in enumerate(zip(x_coords, y_coords)):
        x_vic, y_vic = get_adjacent_points(image, x_c, y_c, size=1)

        markers[y_vic, x_vic] = keypoint_i + 3

    image_grad = compute_image_grad(image)
    mask = watershed(image_grad, markers)

    x = keypoints.x_coords()[0]
    y = keypoints.y_coords()[0]
    mask = mask == mask[y, x]

    return mask


def draw_circle(image_h, image_w, center_x, center_y, radius):
    """
    Creates a circle shaped mask.

    Parameters
    ----------
    image_h: int
        height of the image in pixels
    image_w: int
        width of the image in pixels
    center_x: int
        x_coordinate of circle center
    center_y: int
        y_coordinate of circle center
    radius: int
        radius of the circle

    Returns
    -------
    mask: ndarray of bool
        the resulting mask, the shape is (image_h, image_w)
    """
    coords = np.indices((image_h, image_w))
    coords_x = coords[1]
    coords_y = coords[0]
    mask = (coords_x - center_x) ** 2 + (coords_y - center_y) ** 2 < radius ** 2
    return mask


def generate_contoured_masks(masks, alpha, color=(0, 255, 0), dilate=True):
    """
    Generates the countours of the given masks. Usful for visualization.

    Parameters
    ----------
    masks: ndarray
        masks to draw contours of. The shape sould be (num_masks, image_h, image_w)
    alpha: float
        the transparance of the resulting contours
    color: tuple of int
        color of the masks in RGB representation
    dilate: bool
        whether to dilate the resulting contours

    Returns
    -------
    contoured_masks: ndarray
        the array of masks countours
    """

    masks_contours = np.stack([masks] * 3, axis=-1).astype(np.uint)
    masks_contours = np.copy(masks).astype(np.uint8)

    for i, mask in enumerate(masks_contours):
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask = cv2.drawContours(mask, contours, 0, (255, 0, 0), 1)
        if dilate:
            mask = cv2.dilate(mask, np.array([[1, 1], [1, 1]]))
        masks_contours[i] = mask

    contoured_masks = np.stack(
        [np.zeros_like(masks_contours)] * 3 + [np.zeros_like(masks_contours)], axis=-1
    ).astype(np.uint)

    for color_i in range(3):
        contoured_masks[masks_contours > 1, color_i] = color[color_i]
    contoured_masks[masks_contours > 1, 3] = 255 * alpha
    return contoured_masks


def visualize_masks(image, masks=None, alpha=1.0, dilate=True):
    """
    Visualizes the masks for a given image.

    Parameters
    ----------
    image: ndarray
        image to draw the masks on
    keypoints: KeypointsArray like
        keypoints deifingi the muclei positions. If provided, masks should not be provided
    masks: ndarray
        masks to visualize. The shape must be (num_masks, image_h, image_w).
        If provided, keypoints must not be provided
    alpha: float
        the transparency of the resulting contours
    dilate: bool
        whether to dilate the resulting contours
    """

    contoured_masks = generate_contoured_masks(masks, alpha=alpha, dilate=dilate)

    plt.imshow(image)
    for mask in contoured_masks:
        plt.imshow(mask)
    plt.show()


class NucleiPropagator:
    """
    Propagator, which takes the nuclei keypoints and makes them into the nuclei masks.

    The main idea is using the watershed algorithm on a neighborhood of a given point.
    If the resulting mask is too small or too big, than the propagator tries to choose a LoG filter from
    a predefined set of different sizes and shapes (epllipses). Note, that too large set can result in a
    slow perofrmance.

    Parameters
    ----------
    min_area: int
        minimum area of the watershed mask (all watershed mask with lesser areas will be substituted with a predefined circular mask).
        Not used, if log_filters_cfgs is empty.
    max_area: int
        maximum area of the watershed mask (all watershed mask with greater areas will be substituted with a predefined circular mask).
    average_area: int
        area of circular mask, which substitutes watershed mask if it's aree is too big or to small.
        Not used, if log_filters_cfgs is empty.
    window_size: int
        window size for watershed algoithm.

    """

    def __init__(
        self,
        min_area=6,
        max_area=400,
        average_area=160,
        window_size=30,
    ):
        self.window_size = window_size
        self.min_area = min_area
        self.max_area = max_area
        self.average_area = average_area
        self.average_radius = np.sqrt(average_area / np.pi)

    def generate_clipped_images(self, image, keypoints):
        """
        Generate images clipped around the keypoints

        Parameters
        ----------
        image: ndarray
            input image.
        keypoints: Keypoints
            image keypoints.

        Returns
        -------
            masks: list of ndarray
                list of 2D ndarrays of nuclei masks for the keypoints
            images_clipped: list of nd_array
                list of 2D arrays with clipped images corresponding to each mask
            keypoints_groups: list of Keypoints
                list keypoints for each clipped image. The zeroth keyoint is the base keypoint for the mask
            borders: list of tuple of int
                list of tuples with 4 intergers, encoding the borders of the masks with respect to images
                The order is: bound_x_left, bound_x_right, bound_y_top, bound_y_bottom
        """

        images_clipped = []
        keypoints_groups = []
        borders = []

        if len(keypoints):
            for keypoint_i in range(len(keypoints)):

                image_clipped, keypoints_clipped, borders_clipped = make_window(
                    image, keypoints, keypoint_i, self.window_size
                )
                images_clipped.append(image_clipped)
                keypoints_groups.append(keypoints_clipped)
                borders.append(borders_clipped)

        return images_clipped, keypoints_groups, borders

    def generate_masks(self, image, keypoints, return_area_flags=False):
        """
        Generate masks for image keypoints.

        Parameters
        ----------
        image : ndarray
            input image.
        keypoints : Keypoints
            image keypoints.
        return_area_flags : bool
            if True, the area flags will be returned
            0 means that watershed mask's area was between self.min_area and self.max_area
            1 means, that the watershed mask was smaller that self.min_area
            2 means, that the atershed mask was bigger than self.max_area

        Returns
        -------
        masks: list of ndarray
            list of 2D ndarrays of nuclei masks for the keypoints
        borders: list of tuple of int
            list of tuples with 4 intergers, encoding the borders of the masks with respect to images.
            The order is: bound_x_left, bound_x_right, bound_y_top, bound_y_bottom.
        area_flags: ndarray , optional
            area_flags. Will be return only if return_area_flags is True
        """

        masks = []
        borders = []
        images_clipped = []

        if return_area_flags:
            area_flags = np.zeros(len(keypoints))

        if len(keypoints):

            images_clipped, keypoints_groups, borders = self.generate_clipped_images(
                image, keypoints
            )

            for keypoint_i, (
                image_clipped,
                keypoints_clipped,
                mask_borders,
            ) in enumerate(zip(images_clipped, keypoints_groups, borders)):

                bound_x_left, bound_x_right, bound_y_top, bound_y_bottom = mask_borders
                mask = watershed_mask(image_clipped, keypoints_clipped)

                mask_area = np.sum(mask)
                if mask_area < self.min_area or mask_area > self.max_area:
                    x = keypoints_clipped.x_coords()[0]
                    y = keypoints_clipped.y_coords()[0]
                    mask_h, mask_w = mask.shape
                    mask = draw_circle(mask_h, mask_w, x, y, self.average_radius)

                    if return_area_flags:
                        if mask_area < self.min_area:

                            area_flags[keypoint_i] = 1
                        elif mask_area > self.max_area:

                            area_flags[keypoint_i] = 2

                masks.append(mask)
                images_clipped.append(image_clipped)

        return_tuple = (masks, borders)

        if return_area_flags:
            return_tuple += (area_flags.reshape(-1),)

        return return_tuple

    def masks_to_image_size(self, image, masks, borders):
        """
        Pads the masks to make them maching the image. The masks are usually come from
        generate_masks method.

        Parameters
        ----------
        image: ndarray
            input image.
        masks: list of ndarray
            masks to pad
        borders: list of tuple of int
            list of tuples with 4 intergers, encoding the borders of the masks with respect to images
            The order is: bound_x_left, bound_x_right, bound_y_top, bound_y_bottom

        Returns
        -------
        masks_image_sized : ndarray
            3D ndarray of shape (num_masks, image_h, image_w) with the padded masks
        """
        if len(masks):
            masks_image_sized = np.zeros(
                (len(masks), image.shape[0], image.shape[1])
            ).astype(bool)
            for mask_i, (mask, mask_borders) in enumerate(zip(masks, borders)):
                bound_x_left, bound_x_right, bound_y_top, bound_y_bottom = mask_borders
                mask_ys, mask_xs = np.where(mask)
                mask_xs += bound_x_left
                mask_ys += bound_y_top
                coords_valid = (
                    (mask_xs >= 0)
                    * (mask_xs < image.shape[1])
                    * (mask_ys >= 0)
                    * (mask_ys < image.shape[0])
                )
                mask_xs = mask_xs[coords_valid]
                mask_ys = mask_ys[coords_valid]

                masks_image_sized[mask_i, mask_ys, mask_xs] = True
        else:
            masks_image_sized = np.zeros((1, image.shape[0], image.shape[1])).astype(
                bool
            )

        return masks_image_sized

    def visualize_masks(
        self, image, keypoints=None, masks=None, alpha=1.0, dilate=True
    ):
        """
        Visualizes the masks for a given image. Can generate them from the keypoints,
        or use the pregenerated masks.

        Parameters
        ----------
        image: ndarray
            image to draw the masks on
        keypoints : KeypointsArray like
            keypoints deifingi the muclei positions. If provided, masks should not be provided
        masks: ndarray
            masks to visualise. The shape must be (num_masks, image_h, image_w).
            If provided, keypoints must not be provided
        alpha: float
            the transparance of the resulting contours
        dilate: bool
            whether to dilate the resulting contours
        """

        if keypoints is None and masks is None:
            raise Exception("Either keypoints or masks should be provided")
        if keypoints is not None and masks is not None:
            raise Exception("Keypoints and masks cannot provided  simultaneously")

        if masks is None:
            masks, borders = self.generate_masks(image, keypoints)
            masks = self.masks_to_image_size(image, masks, borders)

        visualize_masks(image, masks=masks, alpha=alpha, dilate=dilate)


class StainAnalyzer:
    """
    Tha analyzer, which takes an image with predicted nuclei potions and estimates the
    DAB stain intensity for each nuclei

    See also
    --------
    pointdet.utils.nucprop.NucleiPropagator
    """

    def __init__(self):

        stain_matrix = np.matrix(
            [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.59, 0.78]]
        )

        self.deconv_matrix = np.linalg.inv(stain_matrix)

        max_matrix = np.copy(self.deconv_matrix)
        max_matrix[self.deconv_matrix < 0] *= 0
        max_matrix[self.deconv_matrix > 0] *= -np.log(0.1 / 255.0)
        max_matrix.sum(axis=1)

        self.dab_max = 1.0
        self.dab_min = 0.0

    def get_stained_image(self, image):
        """
        Applies color deconvolution to the image to extract eozin, hemotoxilin and DAB stains

        Parameters
        ----------
        image: ndarray
            input image. The shape must be (height, width, channels)

        Returns
        -------
        image_stains: ndarray
            the deconvolved image, the shape is the same as image
        """
        image = np.copy(image)
        image[image <= 1.0] = 1.0
        od_image = -np.log(image / 255.0)
        image_stains = np.tensordot(od_image, self.deconv_matrix, axes=((2,), (1,)))

        return image_stains

    def calulate_dab_values(self, image, masks):
        """
        Calculates DAB values for each keypoint on the image.

        To calculate DAB vaules the  following steps are made:

        1) Around each keypoint a nuclei mask is drawn usngg self.propagator
        2) Nuclei mask is applied to the third channel of DAB image
        3) The mean pixel value inside the mask is calculated


        Parameters
        ----------
        image: ndarray
            input image. The shape must be (height, width, channels)

        masks: ndarray
            nuclei masks. The shape should be (num_masks, h, w)

        Returns
        -------
        dab_values: ndarray
            the dab values for the keypoints
        """

        image_stains = self.get_stained_image(image)
        image_dab = image_stains[:, :, 2:]

        dab_values = []
        for mask in masks:

            pixel_values = image_dab[mask]
            mean_value = pixel_values.mean()
            dab_values.append(mean_value)

        dab_values = np.array(dab_values)
        dab_values -= self.dab_min
        dab_values /= self.dab_max - self.dab_min

        return dab_values


    def dab_values_probes(self, image, keypoints, radius):
        """
        Calculates DAB values for the keypoints using probes method:
        for each keypoint a circle of a given radius is taken as a mask.

        Parameters
        ----------
        image: ndarray
            input image, the shape is (H, W, C)
        keypoints: endoanalysis.targets.Keypoints
            keypoints to calculate dab values for
        radius: float
            probe radius

        Returns
        -------
        dab_values: ndarray
            dab values for the keypoints
        """
        image_h, image_w, _ = image.shape
        image_stains = self.get_stained_image(image)
        image_dab = image_stains[:, :, 2:]
        dab_values = np.zeros(len(keypoints))

        for kp_i, (x, y) in enumerate(zip(keypoints.x_coords(), keypoints.y_coords())):
            mask = draw_circle(image_h, image_w, x, y, radius)
            pixel_values = image_dab[mask]
            dab_values[kp_i]= pixel_values.mean()

        return dab_values
