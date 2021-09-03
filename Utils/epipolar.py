import random
from Utils import utils
import cv2 as cv
import numpy as np


random.seed(19)


def plot_epilines(image, points, epipole, color=None, thickness=3):
    epipole = epipole.reshape(-1)
    lines = []
    for point in points:
        lines.append(utils.vector_outer_product(utils.to_projective(point),
                                                utils.to_projective(epipole)))

    im = image.copy()
    for i in range(len(lines)):
        line = lines[i]
        x0 = 0
        y0 = int(-line[2]/line[1])
        x1 = int(im.shape[1] - 1)
        y1 = int(-line[2]/line[1] - x1 * line[0]/line[1])
        if color is None:
            c = np.random.randint(0, 255, 3).tolist()
        else:
            c = color
        im = cv.line(im, (x0, y0), (x1, y1), c, thickness)
    return im
