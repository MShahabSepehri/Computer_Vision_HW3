import cv2 as cv
import numpy as np


def get_sift_key_points(image):
    sift = cv.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(image, None)
    return keyPoints, descriptors


def get_match_points(des1, des2, ratio_tr):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches_alpha1 = bf.knnMatch(des1, des2, k=2)
    matches1 = []
    for p1, p2 in matches_alpha1:
        if p1.distance < ratio_tr * p2.distance:
            matches1.append(p1)

    matches_alpha2 = bf.knnMatch(des2, des1, k=2)
    matches2 = []
    for p1, p2 in matches_alpha2:
        if p1.distance < ratio_tr * p2.distance:
            matches2.append(p1)

    matches = []
    for m1 in matches1:
        q1 = m1.queryIdx
        t1 = m1.trainIdx
        for m2 in matches2:
            if t1 == m2.queryIdx and q1 == m2.trainIdx:
                matches.append(m1)
                break
    return matches


def get_corresponding_points(img1, img2, ratio_tr):
    kps1, des1 = get_sift_key_points(img1)
    kps2, des2 = get_sift_key_points(img2)

    matches = get_match_points(des1, des2, ratio_tr)

    srcPoints = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    desPoints = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return srcPoints, desPoints
