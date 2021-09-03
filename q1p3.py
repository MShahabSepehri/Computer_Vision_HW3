import numpy as np
import cv2 as cv
from Utils import vanishing, utils
from Utils import configReader as cr


pathDict = cr.get_config('Gen', 'path')
dataPath = pathDict['data']

image = utils.get_image(dataPath + "vns.jpg")

Vx = np.array([9.45983319e+03, 2.63725943e+03, 1.00000000e+00])
Vy = np.array([-2.38315041e+04, 3.86043297e+03, 1.00000000e+00])
Vz = np.array([-2.20025854e+03, -1.08724706e+05, 1.00000000e+00])

P, f = vanishing.get_P_and_f(Vx, Vy, Vz)

angle = 2.10/180*np.pi
Rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
angle = 7.18/180*np.pi
Rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

k = np.array([[f, 0, P[0]], [0, f, P[1]], [0, 0, 1]])
kInv = np.linalg.inv(k)

offset = np.array([[1, 0, 50], [0, 1, 1850], [0, 0, 1]])
H = np.matmul(k, np.matmul(np.matmul(Rx, Rz), kInv))
print('desired homography matrix:')
print(H)
H = np.matmul(offset, H)
resultImage = cv.warpPerspective(image, H, (4300, 3000))

utils.plot_array('res04.jpg', resultImage)
