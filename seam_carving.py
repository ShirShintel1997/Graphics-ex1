from typing import Dict, Any

import numpy as np

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


def vertical_resize(image: NDArray, k: int, forward_implementation: bool) -> Dict[str, NDArray]:
    grey_image = utils.to_grayscale(image)
    seams_mask = np.ones_like(image)
    indices = create_indices_matrix(len(image), len(image[0]))
    for i in range(k) :
        cost_matrix = calculate_cost_matrix(grey_image)
        seam_mask = find_best_seam(cost_matrix, indices)
        remove_seam(grey_image, seam_mask)
        remove_seam(indices, seam_mask)
    # update original image
    # visualize



def create_indices_matrix(rows, cols):
    return [[x for x in range(cols)] for x in range(rows)]

def calculate_cost_matrix(image: NDArray):
    gradients = utils.get_gradients(image)
    cost_matrix = np.zeros_like(gradients)
    for row in range(len(cost_matrix)):
        for col in range(len(cost_matrix[0])):
            if row == 0:
                cost_matrix[row, col] = gradients[row,col]
            if col == 0:
                cost_matrix[row, col] = gradients[row, col] + \
                                        np.min(cost_matrix[row - 1, col], cost_matrix[row - 1, col + 1])
            if col == len(cost_matrix[0]) -1 :
                cost_matrix[row, col] = gradients[row, col] + \
                                        np.min(cost_matrix[row - 1, col - 1], cost_matrix[row - 1, col])
            cost_matrix[row,col] = gradients[row,col] + \
            np.min(cost_matrix[row-1, col-1], cost_matrix[row-1, col], cost_matrix[row-1, col+1])
    return cost_matrix


def find_best_seam(cost_matrix: NDArray, indices: NDArray, seams_mask: NDArray):
    current_seam_mask = np.ones_like(cost_matrix)
    row = len(cost_matrix) - 1
    cols = len(cost_matrix[0]) -1
    col = 0
    while row >= 0 :
        if row < len(cost_matrix)-1:
            min_cost_ind = np.argmin(cost_matrix[row-1, np.max(col-1,0): np.min(col+2, cols)])
        else :
            min_cost_ind = np.argmin(cost_matrix[row])
        current_seam_mask[row, min_cost_ind]
        seams_mask[row, indices[min_cost_ind]] = 0
        row -= 1
        col = min_cost_ind
    return current_seam_mask

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
