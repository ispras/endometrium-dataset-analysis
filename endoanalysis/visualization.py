import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches



def visualize_keypoints(
    image, 
    keypoints, 
    class_colors, 
    labels=None, 
    fig=None,
    ax=None, 
    circles_kwargs={"radius": 3, "alpha": 1.0}
    ):
    """
    Visualise keypoints on the image

    Parameters
    ----------
    image: ndarray
        input image. The shape must be (H, W, C)
    keypoints: endoanalysis.targets.keypoints
        keypoints to visualize
    class_colors:
        dictionary of colours for different classes
    labels: iterble of str
        texts to label the keypoints
    ax: matplotlib.axes._subplots.AxesSubplot
        plt subplot to draw image and keypoints.
        If not provided, will be generated and returned
    circles_kwargs: dict
        kwargs for circles pathces indicating the keypints

    Returns
    -------
    fig: matplotlib.figure.Figure or None
        plt figure object. if ax parametero is not provided, None will be returned
    ax: matplotlib.axes._subplots.AxesSubplot
        plt axis object with image and keypoints

    Example
    -------
    >>> visualize_keypoints(
    ...     image, 
    ...     keypoints_image, 
    ...     class_colors= {x: cm.Set1(x) for x in range(10)},
    ...     fig=fig,
    ...     ax=ax, 
    ...     circles_kwargs={"radius": 2.5, "alpha": 1.0,  "linewidth": 2, 'ec': (0,0,0)}
    ...     )
    """
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = None
   
    ax.imshow(image)
    ax.autoscale(False)

    x_coords = keypoints.x_coords()
    y_coords = keypoints.y_coords()
    classes = keypoints.classes() 
    for i, (center_x, center_y, obj_class) in enumerate(zip(x_coords, y_coords, classes)):
             

        patch = patches.Circle((center_x, center_y), color=class_colors[obj_class], **circles_kwargs)
        ax.add_patch(patch)

        if labels is not None:
           ax.text(
                center_x ,
                center_y ,
                labels[i],
                c="b",
                fontweight="semibold",
            ) 
    

    return fig, ax


def generate_contoured_masks( masks, alpha, color=(0,255,0), dilate = True):
    """
    Generates the countours of the given masks. Usful for visualization.

    Parameters
    ----------
    masks : ndarray
        masks to draw contours of. The shape sould be (num_masks, image_h, image_w)
    alpha : float
        the transparance of the resulting contours
    color: tuple of int
        color of the masks in RGB representation
    dilate : bool
        whether to dilate the resulting contours

    Returns
    -------
    contoured_masks : ndarray
        the array of masks countours
    """

    masks_contours = np.stack([masks]*3 , axis=-1).astype(np.uint)
    masks_contours = np.copy(masks).astype(np.uint8)


    for i, mask in enumerate(masks_contours):
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        mask = cv2.drawContours(mask, contours, 0, (255,0,0), 1)
        if dilate:
            mask = cv2.dilate(mask, np.array([[1,1],[1,1]]))
        masks_contours[i] = mask


    contoured_masks = np.stack([np.zeros_like(masks_contours)]*3 +[np.zeros_like(masks_contours)], axis=-1).astype(np.uint)

    for color_i in range(3):
        contoured_masks[masks_contours > 1, color_i] = color[color_i]
    contoured_masks[masks_contours > 1, 3] = 255 * alpha
    return contoured_masks


def visualize_masks(image, masks=None, alpha=1., dilate=True):
    """
    Visualizes the masks for a given image. 

    Parameters
    ----------
    image : ndarray
        image to draw the masks on
    keypoints : KeypointsArray like
        keypoints deifingi the muclei positions. If provided, masks should not be provided
    masks : ndarray
        masks to visualize. The shape must be (num_masks, image_h, image_w). 
        If provided, keypoints must not be provided
    alpha : float
        the transparency of the resulting contours
    dilate : bool
        whether to dilate the resulting contours
    """

    contoured_masks = generate_contoured_masks(masks, alpha=alpha, dilate = dilate)

    plt.imshow(image) 
    for mask in contoured_masks:
        plt.imshow(mask)
    plt.show()