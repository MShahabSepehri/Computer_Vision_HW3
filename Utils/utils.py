import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from Utils import configReader as cr

pathDict = cr.get_config('Gen', 'path')
resultsPath = pathDict['results']


def get_image(path, convert=True):
    if convert:
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    else:
        return cv.imread(path)


def convert_from_projective(point):
    p = point.reshape(-1)
    x = p[0] / p[2]
    y = p[1] / p[2]
    return np.array([[x, y]]).transpose()


def check_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)


def plot_array(name, array):
    check_dir(resultsPath)
    cv.imwrite(resultsPath + name, cv.cvtColor(array.astype(np.uint8), cv.COLOR_RGB2BGR))


def vector_outer_product(vec1, vec2, tr=None):
    product = np.zeros(vec1.shape)
    product[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    product[1] = - vec1[0] * vec2[2] + vec1[2] * vec2[0]
    product[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if product[2] != 0:
        ans = product / product[2]
        if tr is None:
            return ans
        else:
            if np.max(np.abs(ans)) < tr:
                return ans


def get_dist(vec1, vec2):
    maxCoor = max(np.max(np.abs(vec1)), np.max(np.abs(vec2)))
    if maxCoor == 0:
        maxCoor = 1
    return np.sqrt(np.sum((vec1 / maxCoor - vec2 / maxCoor) ** 2)) * maxCoor


def add_border(image, d=(30, 30, 30, 30), color=(0, 0, 0)):
    return cv.copyMakeBorder(image, d[0], d[1], d[2], d[3], cv.BORDER_CONSTANT, value=color)


def plot_double_arrays(name, image1, image2, d=(30, 30, 30, 30), borderColor=(0, 0, 0), horizontal=True):
    image1 = add_border(image1, d, borderColor)
    image2 = add_border(image2, d, borderColor)

    if horizontal:
        image = cv.hconcat((image1, image2))
    else:
        image = cv.vconcat((image1, image2))
    plot_array(name, image)


def plot_points(image, points, color, radius=7):
    copyImage = image.copy()
    for point in points:
        copyImage = cv.circle(copyImage, (int(point[0]), int(point[1])), radius, color, -1)
    return copyImage


def plot_not_in_image_point(image, point, color, radius=20):
    xMin, yMin, xMax, yMax = image_with_outside_points_size_coor(image, [point], radius)
    out = np.zeros((yMax + yMin + 1, xMax + xMin + 1, 3), dtype=np.uint8)
    out[yMin: image.shape[0] + yMin, xMin: image.shape[1] + xMin, :] = image
    return cv.circle(out, (int(point[0] + xMin), int(point[1] + yMin)), radius, color, -1)


def image_with_outside_points_size_coor(image, points, radius=20):
    xMin = 0
    yMin = 0
    xMax = image.shape[1] - 1
    yMax = image.shape[0] - 1
    for point in points:
        x = point[0]
        y = point[1]
        xMin = int(min(x - radius, xMin))
        yMin = int(min(y - radius, yMin))
        xMax = int(max(x + radius, xMax))
        yMax = int(max(y + radius, yMax))
    return -xMin, -yMin, xMax, yMax


def save_fig(name):
    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)
    plt.savefig(resultsPath + name, bbox_inches='tight')


def normalize_data(data, scale0=2):
    dSum = np.zeros(data[0].shape)
    for point in data:
        dSum += point
    mean = dSum / len(data)

    scale = 0
    for point in data:
        diff = point - mean
        scale += np.sqrt(np.sum(diff ** 2)) / scale0

    normalizedData = [(point - mean) / scale for point in data]
    T = np.array([[1 / scale, 0, -mean[0] / scale], [0, 1 / scale, -mean[1] / scale], [0, 0, 1]])
    return normalizedData, T


def replace_rows(mat, r1, r2):
    out = mat.copy()
    tmp = out[r1, :].copy()
    out[r1, :] = out[1, :]
    out[r2, :] = tmp
    return out


def load_data_with_resize(path, size):
    image = get_image(path, convert=False)
    image = cv.resize(image, size, interpolation=cv.INTER_AREA)
    return image.reshape(-1)


def load_all_data_as_vector(path, size):
    data = []
    label = []
    for subdir, _, files in os.walk(path):
        for file in files:
            data.append(load_data_with_resize(os.path.join(subdir, file), size))
            label.append(subdir.replace(path, ''))
    return data, label


def to_projective(vec):
    return np.array([vec[0], vec[1], 1])
