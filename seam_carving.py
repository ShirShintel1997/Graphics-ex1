from typing import Dict, Any

import utils
import numpy as np
from PIL import Image

NDArray = Any

VERTICAL_SEAM_COLOUR = np.array([255, 0, 0])
HORIZONTAL_SEAM_COLOUR = np.array([0, 0, 0])


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respectively).
    """
    height, width = image.shape[:2]
    vert_resize, vert_mask = vertical_resize(image, out_width - width)
    horiz_resize, horiz_mask = vertical_resize(vert_resize.rotate(), out_height - height)
    vertical_seams = get_visualization(vert_resize, vert_mask, True)
    horizontal_seams = get_visualization(horiz_resize, horiz_mask, False)
    return { 'resized' : get_visualization(horiz_resize), 'vertical_seams' : vertical_seams ,'horizontal_seams' : horizontal_seams}
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


def vertical_resize(image: NDArray, k: int, forward_implementation: bool) -> [NDArray, Any]:
    grey_image = utils.to_grayscale(image)
    seams_mask = np.ones_like(image)
    indices = create_indices_matrix()
    for i in range(k) :
        cost_matrix = calculate_cost_matrix(grey_image)
        seam_mask = find_best_seam(cost_matrix, indices)
        remove_seam(grey_image, seam_mask)
        remove_seam(indices, seam_mask)
    return get_new_image(image, seams_mask, k), seams_mask

def create_indices_matrix():
    raise NotImplementedError('You need to implement this!')

def calculate_M():
    raise NotImplementedError('You need to implement this!')

def find_best_seam(img, indices):
    raise NotImplementedError('You need to implement this!')

def remove_seam(img, seam_mask):
    height, width = img.shape[:2]
    return img[seam_mask].reshape((height, width - 1))

def get_new_image(img, img_mask, k):
    if k > 0:
        return insert_seams(img, img_mask, k)
    return remove_seams(img, img_mask, k)

def insert_seams(img, img_mask, k):
    height, width = img.shape[:2]
    new_img = np.zeros((height, width + k, 3))
    for row in range(height):
        col_idx = 0
        for col in range(width):
            new_img[row, col_idx] = img[row, col]
            if img_mask[row, col] is False:
                col_idx += 1
                new_img[row, col_idx] = img[row, col]
            col_idx += 1
    return new_img

def remove_seams(img, img_mask, k):
    height, width, c = img.shape
    new_img = img[img_mask].reshape(height, width - k)
    return new_img

def update_indices():
    raise NotImplementedError('You need to implement this!')

def get_visualization(img, img_mask=None, is_vertical=True):
    visual = img.astype('uint8')
    if img_mask is not None:
        seam_colour = VERTICAL_SEAM_COLOUR if is_vertical else HORIZONTAL_SEAM_COLOUR
        visual[np.where(img_mask == False)] = seam_colour
    pil_img = Image.fromarray(visual, 'RGB')
    # pil_img.show()
    return pil_img
