from typing import Dict, Any

import utils
import numpy as np
from PIL import Image

NDArray = Any

VERTICAL_SEAM_COLOUR = np.array([255, 0, 0])
HORIZONTAL_SEAM_COLOUR = np.array([0, 0, 0])
EPSILON = 0.000000000001


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
    vert_resize, vert_mask = vertical_resize(image, out_width - width, forward_implementation)
    horiz_resize, horiz_mask = vertical_resize(np.rot90(np.copy(vert_resize)), out_height - height,
                                               forward_implementation)
    vertical_seams = get_visualization(image, vert_mask, True)
    horizontal_seams = get_visualization(np.rot90(np.copy(vert_resize)), horiz_mask, False)
    return {'resized': np.rot90(np.rot90(np.rot90(horiz_resize))), 'vertical_seams': vertical_seams,
            'horizontal_seams': np.rot90(np.rot90(np.rot90(horizontal_seams)))}
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


def vertical_resize(image: NDArray, k: int, forward_implementation: bool):
    gray_image = utils.to_grayscale(image)
    indices = create_indices_matrix(len(image), len(image[0]))
    gradients = utils.get_gradients(image)
    seams_mask = np.ones_like(gradients)
    for i in range(np.abs(k)):
        if forward_implementation:
            cost_matrix = calculate_forward_cost_matrix(gray_image, gradients)
            seam_mask = find_best_forward_seam(gray_image, cost_matrix, indices, gradients, seams_mask)
            gray_image = remove_seam(gray_image, seam_mask)
        else:
            cost_matrix = calculate_cost_matrix(gradients)
            seam_mask = find_best_seam(cost_matrix, indices, seams_mask)
        gradients = remove_seam(gradients, seam_mask)
        indices = remove_seam(indices, seam_mask)
    seams_mask = seams_mask.astype(bool)
    new_image = get_new_image(image, seams_mask, k)
    return new_image, seams_mask


def create_indices_matrix(rows, cols):
    return np.array([[x for x in range(cols)] for y in range(rows)])


def calculate_cost_matrix(gradients: NDArray):
    cost_matrix = np.zeros_like(gradients).astype(float)
    for row in range(len(cost_matrix)):
        for col in range(len(cost_matrix[0])):
            if row == 0:
                cost_matrix[row, col] = gradients[row, col]
            elif col == 0:
                cost_matrix[row, col] = gradients[row, col] + \
                                        min(cost_matrix[row - 1, col], cost_matrix[row - 1, col + 1])
            elif col == len(cost_matrix[0]) - 1:
                cost_matrix[row, col] = gradients[row, col] + \
                                        min(cost_matrix[row - 1, col - 1], cost_matrix[row - 1, col])
            else:
                cost_matrix[row, col] = gradients[row, col] + \
                                        min(cost_matrix[row - 1, col - 1], cost_matrix[row - 1, col],
                                            cost_matrix[row - 1, col + 1])
    return cost_matrix


def calculate_forward_cost_matrix(img: NDArray, gradients: NDArray):
    cost_matrix = np.zeros_like(img)
    for row in range(len(cost_matrix)):
        for col in range(len(cost_matrix[0])):
            if row == 0:
                cost_matrix[row, col] = gradients[row, col]
            elif col == 0:
                cost_matrix[row, col] = gradients[row, col] + min(cost_matrix[row - 1, 0] + calc_cv(img, row, 0),
                                                                  cost_matrix[row - 1, 1] + calc_cr(img, row, 0))
            elif col == len(cost_matrix[0]) - 1:
                cost_matrix[row, col] = gradients[row, col] + min(
                    cost_matrix[row - 1, col - 1] + calc_cl(img, row, col),
                    cost_matrix[row - 1, col] + calc_cv(img, row, col))
            else:
                cost_matrix[row, col] = gradients[row, col] + min(
                    cost_matrix[row - 1, col - 1] + calc_cl(img, row, col),
                    cost_matrix[row - 1, col] + calc_cv(img, row, col),
                    cost_matrix[row - 1, col + 1] + calc_cr(img, row, col))
    return cost_matrix


def calc_cl(img: NDArray, i: int, j: int):
    if j == 0:
        return np.abs(img[i, j+1]-255) + np.abs(img[i-1, j]-255)
    if j == len(img[0]) - 1 :
        return np.abs(255-img[i,j-1]) + np.abs(img[i-1, j]-img[i,j-1])
    return np.abs(img[i, j+1]-img[i,j-1]) + np.abs(img[i-1, j]-img[i,j-1])

def calc_cv(img: NDArray, i: int, j: int) :
    if j == 0:
        return np.abs(img[i, j+1]-255)
    if j == len(img[0]) - 1 :
        return np.abs(255-img[i,j-1])
    return np.abs(img[i, j+1]-img[i,j-1])

def calc_cv(img: NDArray, i: int, j: int):
    if j == 0:
        return np.abs(img[i, j+1]-255) + np.abs(img[i-1, j]-img[i,j+1])
    if j == len(img[0]) - 1 :
        return np.abs(255-img[i,j-1]) + np.abs(img[i-1, j]-255)
    return np.abs(img[i, j+1]-img[i,j-1]) + np.abs(img[i-1, j]-img[i,j+1])

def calc_cr(img: NDArray, i: int, j: int):
    if j == 0:
        return np.abs(img[i, j + 1] - 255) + np.abs(img[i - 1, j] - img[i, j + 1])
    if j == len(img[0]) - 1:
        return np.abs(255 - img[i, j - 1]) + np.abs(img[i - 1, j] - 255)
    return np.abs(img[i, j + 1] - img[i, j - 1]) + np.abs(img[i - 1, j] - img[i, j + 1])


def find_best_seam(cost_matrix: NDArray, indices: NDArray, seams_mask: NDArray):
    current_seam_mask = np.ones_like(cost_matrix)
    row = len(cost_matrix) - 1
    cols = len(cost_matrix[0]) - 1
    col = 0
    while row >= 0 :
        if row < len(cost_matrix) - 1: #when not in last row
            min_cost_ind = col - 1 +np.argmin(cost_matrix[row, max(col - 1, 0): min(col + 2, cols)])
            if col == 0: # in this case min_cost_ind == -1 or 0, should be 0 or 1
                min_cost_ind += 1
            elif min_cost_ind > cols:
                min_cost_ind = cols
        else:  # when in last row:
            min_cost_ind = np.argmin(cost_matrix[row])
        current_seam_mask[row, min_cost_ind] = 0
        seams_mask[row, indices[row, min_cost_ind]] = 0
        row = row - 1
        col = min_cost_ind
    return current_seam_mask


def find_best_forward_seam(img: NDArray, cost_matrix: NDArray, indices: NDArray, gradients: NDArray,
                           seams_mask: NDArray):
    current_seam_mask = np.ones_like(cost_matrix)
    row = len(cost_matrix) - 1
    cols = len(cost_matrix[0]) - 1
    col = 0
    while row >= 0 :
        if row < len(cost_matrix)-1 and row > 0:
            if abs(cost_matrix[row,col] - (gradients[row,col] + cost_matrix[row-1,col]+ calc_cv(img, row, col)))<=EPSILON:
                min_cost_ind = col
            elif abs(cost_matrix[row,col] - (gradients[row,col] + cost_matrix[row-1,col-1]+ calc_cl(img, row, col)))<=EPSILON:
                min_cost_ind = col - 1
            else:
                min_cost_ind = col + 1
        elif row == 0:
            min_cost_ind = col - 1 + np.argmin(cost_matrix[row, max(col - 1, 0): min(col + 2, cols)])
            if col == 0:  # in this case min_cost_ind == -1 or 0, should be 0 or 1
                min_cost_ind += 1
        else :
            min_cost_ind = np.argmin(cost_matrix[row])

        # TODO: Asif added the next 2 lines trying to solve the problem with forward_implementation but there still is a problem
        if min_cost_ind == seams_mask.shape[0]:
            min_cost_ind -= 1
        current_seam_mask[row, min_cost_ind] = 0
        seams_mask[row, indices[row,min_cost_ind]] = 0
        row -= 1
        col = min_cost_ind
    return current_seam_mask


def remove_seam(img, seam_mask):
    height, width = img.shape[:2]
    new_img = img[seam_mask != 0].reshape(height, width - 1)
    return new_img


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
            if img_mask[row, col] == False:
                col_idx += 1
                new_img[row, col_idx] = img[row, col]
            col_idx += 1
    return new_img


def remove_seams(img, img_mask, k):
    height, width, c = img.shape
    return img[np.where(img_mask == True)].reshape(height, width - abs(k), c)


def get_visualization(img, img_mask=None, is_vertical=True):
    if img_mask is not None:
        seam_colour = VERTICAL_SEAM_COLOUR if is_vertical else HORIZONTAL_SEAM_COLOUR
        img[np.where(img_mask == False)] = seam_colour
    return img
