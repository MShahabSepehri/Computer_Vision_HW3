from Utils import utils, matching, epipolar
from Utils import configReader as cr
import cv2 as cv
import numpy as np


params = cr.get_config('Q2', 'params')
pathDict = cr.get_config('Gen', 'path')
dataPath = pathDict['data']

image1 = utils.get_image(dataPath + "01.JPG")
image2 = utils.get_image(dataPath + "02.JPG")

ratio_tr = params['ratio_tr']
keyPoints1, keyPoints2 = matching.get_corresponding_points(image1, image2, ratio_tr)

F, mask = cv.findFundamentalMat(keyPoints1, keyPoints2, cv.FM_RANSAC)
print("Fundamental matrix:")
print(F)
print('\n')

inliers1 = keyPoints1[mask == 1]
inliers2 = keyPoints2[mask == 1]
outliers1 = keyPoints1[mask == 0]
outliers2 = keyPoints2[mask == 0]

im1 = utils.plot_points(image1, inliers1, (0, 255, 0))
im1 = utils.plot_points(im1, outliers1, (255, 0, 0))
im2 = utils.plot_points(image2, inliers2, (0, 255, 0))
im2 = utils.plot_points(im2, outliers2, (255, 0, 0))

utils.plot_double_arrays("res05.jpg", im1, im2)

_, _, vh = np.linalg.svd(F)
eProjective = vh[-1, :].reshape(3, 1)

_, _, vh = np.linalg.svd(F.transpose())
ePrimeProjective = vh[-1, :].reshape(3, 1)

e = utils.convert_from_projective(eProjective)
ePrime = utils.convert_from_projective(ePrimeProjective)

print("e:")
print(e)
print('\n')
print("e_prime:")
print(ePrime)

utils.plot_array("res06.jpg", utils.plot_not_in_image_point(image1, e, (0, 128, 128), radius=100))
utils.plot_array("res07.jpg", utils.plot_not_in_image_point(image2, ePrime, (0, 128, 128), radius=100))

points1 = inliers1[:10]
points2 = inliers2[:10]

image1_with_epilines = epipolar.plot_epilines(image1, points1, e)
image1_with_epilines = utils.plot_points(image1_with_epilines, points1, (0, 0, 255), radius=15)
image2_with_epilines = epipolar.plot_epilines(image2, points2, ePrime)
image2_with_epilines = utils.plot_points(image2_with_epilines, points2, (0, 0, 255), radius=15)

utils.plot_double_arrays("res08.jpg", image1_with_epilines, image2_with_epilines)
