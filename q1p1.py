import cv2 as cv
from Utils import configReader as cr
from Utils import utils, vanishing
import numpy as np

pathDict = cr.get_config('Gen', 'path')
dataPath = pathDict['data']

image = utils.get_image(dataPath + "vns.jpg")
deg = np.pi/180

projectiveLines = vanishing.get_lines(image, 90*deg, 105*deg, 500, 700, hough_tr=200)
Vx, _, _ = vanishing.get_the_best_intersection(projectiveLines, 50, 100000, numLines=3)

projectiveLines = vanishing.get_lines(image, 85*deg, 90*deg, 200, 400, hough_tr=400)
Vy, _, _ = vanishing.get_the_best_intersection(projectiveLines, 50, 20000, numLines=3)

projectiveLines = vanishing.get_lines(image, -5*deg, 5*deg, 600, 700, hough_tr=500)
Vz, _, _ = vanishing.get_the_best_intersection(projectiveLines, 50, 10000, numLines=3)

print("Vx: ")
print(Vx)
print("Vy: ")
print(Vy)
print("Vz: ")
print(Vz)

h = utils.vector_outer_product(Vx, Vy)
h /= np.sqrt(np.sum(h**2) - 1)

print("h: ")
print(h)

shift = 500
im1 = np.zeros((image.shape[0] + shift, image.shape[1] + shift, 3), dtype=np.uint8)
im1[:image.shape[0], :image.shape[1], :] = image
im1 = cv.line(im1, (int(Vx[0]), int(Vx[1])), (int(Vy[0]), int(Vy[1])), (255, 0, 0), 10)

utils.plot_array('res01.jpg', im1)

xMin, yMin, xMax, yMax = utils.image_with_outside_points_size_coor(image, [Vx, Vy, Vz], radius=100)
xMin = int(xMin/2)
yMin = int(yMin/2)
xMax = int(xMax/2)
yMax = int(yMax/2)

Vxs = (int(Vx[0]/2 + xMin), int(Vx[1]/2 + yMin))
Vys = (int(Vy[0]/2 + xMin), int(Vy[1]/2 + yMin))
Vzs = (int(Vz[0]/2 + xMin), int(Vz[1]/2 + yMin))

im1 = np.zeros((yMin + yMax + 1, xMin + xMax + 1, 3), dtype=np.uint8) + 255
im1[yMin: int(image.shape[0]/2) + yMin, xMin: int(image.shape[1]/2) + xMin, :] = 0
d = 50
im1[yMin + d: int(image.shape[0]/2) + yMin - d, xMin + d: int(image.shape[1]/2) + xMin - d, :] = 255
im1 = cv.circle(im1, Vxs, 100, (0, 0, 255), -1)
im1 = cv.circle(im1, Vys, 100, (0, 255, 0), -1)
im1 = cv.circle(im1, Vzs, 100, (255, 0, 0), -1)
im1 = cv.line(im1, Vxs, Vys, (255, 0, 0), 10)

utils.plot_array('res02.jpg', im1)
