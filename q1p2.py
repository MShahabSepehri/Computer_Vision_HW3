import cv2 as cv
from Utils import configReader as cr
from Utils import utils, vanishing
import numpy as np
from scipy.spatial.transform import Rotation


pathDict = cr.get_config('Gen', 'path')
dataPath = pathDict['data']

image = utils.get_image(dataPath + "vns.jpg")
Vx = np.array([9.45983319e+03, 2.63725943e+03, 1.00000000e+00])
Vy = np.array([-2.38315041e+04, 3.86043297e+03, 1.00000000e+00])
Vz = np.array([-2.20025854e+03, -1.08724706e+05, 1.00000000e+00])
h = np.array([-3.67167220e-02, -9.99325714e-01, 2.98281523e+03])

P, f = vanishing.get_P_and_f(Vx, Vy, Vz)

print("f: {0:.2f}".format(f))
print("P:")
print(P)

color = (255, 255, 255)
im1 = image.copy()
im1 = cv.circle(im1, (int(P[0]), int(P[1])), 40, (255, 0, 0), -1)
header = np.zeros((500, im1.shape[1], 3), dtype=np.uint8)
header[:, :, 0] = color[0]
header[:, :, 1] = color[1]
header[:, :, 2] = color[2]

im1WithTitle = cv.vconcat((header, im1))
font = cv.FONT_HERSHEY_DUPLEX
im1WithTitle = cv.putText(im1WithTitle, "f = " + '{:.2f}'.format(f), (750, 450), font, 12, (0, 0, 0), 10, 0)

utils.plot_array('res03.jpg', im1WithTitle)

k = np.array([[f, 0, P[0]], [0, f, P[1]], [0, 0, 1]])
kInv = np.linalg.inv(k)
B = np.matmul(kInv.transpose(), kInv)
l1 = 1/np.sqrt(np.matmul(Vx.transpose(), np.matmul(B, Vx)))
l2 = 1/np.sqrt(np.matmul(Vy.transpose(), np.matmul(B, Vy)))
l3 = 1/np.sqrt(np.matmul(Vz.transpose(), np.matmul(B, Vz)))

r = np.zeros((3, 3))
r[:, 0] = l1*Vx
r[:, 1] = l2*Vy
r[:, 2] = l3*Vz

R = np.matmul(kInv, r)

goodR = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
myRot = np.matmul(goodR, np.linalg.inv(R))

rot = Rotation.from_matrix(myRot)
angles = rot.as_euler('YXZ', degrees=True)

print("Angle of rotation around Z axis: {0:.2f} degrees".format(angles[2]))
print("Angle of rotation around X axis: {0:.2f} degrees".format(angles[1]))
