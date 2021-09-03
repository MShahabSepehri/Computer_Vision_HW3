import numpy as np
import cv2 as cv
import math
import random


def get_lines(image, min_theta, max_theta, canny_min_val=500, canny_max_val=600,
              hough_tr=150, rho=1, theta=math.pi/180):

    imageEdge = cv.Canny(cv.cvtColor(image.copy(), cv.COLOR_RGB2GRAY), canny_min_val, canny_max_val)
    lines = cv.HoughLines(imageEdge, rho, theta, hough_tr, min_theta=min_theta, max_theta=max_theta)
    projectiveLines = []

    for i in range(lines.shape[0]):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        projectiveLines.append(np.array([[-np.cos(theta), -np.sin(theta), rho]]).transpose())
    return projectiveLines


def get_the_best_intersection_mse(projectiveLines):
    A = np.zeros((len(projectiveLines), 3))

    for i in range(len(projectiveLines)):
        A[i, :] = projectiveLines[i].reshape(-1)
    _, _, vh = np.linalg.svd(A)
    V = vh[-1, :].reshape(3, 1)
    V /= V[2]
    return V


def get_the_best_intersection(projectiveLines, tr, N, numLines=3):
    if numLines is None:
        return get_the_best_intersection_mse(projectiveLines), None, None

    maxSupport = 0
    i = 0
    while i < N:
        i += 1
        lines = random.sample(projectiveLines, numLines)
        A = np.zeros((numLines, 3))
        for j in range(numLines):
            A[j, :] = lines[j].reshape(-1)
        _, _, vh = np.linalg.svd(A)
        V = vh[-1, :].reshape(3, 1)
        if V[2] == 0:
            i -= 1
            continue
        V /= V[2]
        support = 0
        mask = []
        for line in projectiveLines:
            if np.abs(np.sum(line*V)) < tr:
                support += 1
                mask.append(1)
            else:
                mask.append(0)
        if support > maxSupport:
            maxSupport = support
            bestPoint = V

    return bestPoint.reshape(-1), mask, maxSupport


def get_P_and_f(Vx, Vy, Vz):
    a1 = Vx[0]
    b1 = Vx[1]
    a2 = Vy[0]
    b2 = Vy[1]
    a3 = Vz[0]
    b3 = Vz[1]

    a = np.array([[a1 - a3, b1 - b3], [a2 - a3, b2 - b3]])
    b = np.array([a2 * (a1 - a3) + b2 * (b1 - b3), a1 * (a2 - a3) + b1 * (b2 - b3)])

    P = np.linalg.solve(a, b)
    px = P[0]
    py = P[1]

    f = np.sqrt(-px ** 2 - py ** 2 + (a1 + a2) * px + (b1 + b2) * py - (a1 * a2 + b1 * b2))
    return P, f
