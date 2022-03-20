from typing import Dict, Any

import utils

NDArray = Any


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
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    raise NotImplementedError('You need to implement this!')
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


def backward_resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    grey_image = utils.to_grayscale(image)
    gradients = utils.get_gradients(image)
    seams = []


def create_indices_matrix():
    raise NotImplementedError('You need to implement this!')

def calculate_M():
    raise NotImplementedError('You need to implement this!')

def find_best_seam():
    raise NotImplementedError('You need to implement this!')

def remove_seam():
    raise NotImplementedError('You need to implement this!')

def duplicate_seam():
    raise NotImplementedError('You need to implement this!')

def update_indices():
    raise NotImplementedError('You need to implement this!')

def visualize_vertically():
    raise NotImplementedError('You need to implement this!')

def visualize_horizontally():
    raise NotImplementedError('You need to implement this!')

def resize_height():
    raise NotImplementedError('You need to implement this!')
