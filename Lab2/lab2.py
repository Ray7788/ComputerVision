import cv2 as cv
from scipy.ndimage import sobel, gaussian_filter, convolve, maximum_filter
import numpy as np


SIGMA = 0.5
GAUSSIAN_SIZE = 5
ALPHA = 0.05
THRESHOLD = 1e7
RATIO = 0.7
LIMIT = 50


# 1. Feature detection
def harrisPointsDetector(mat, threshold=THRESHOLD, alpha=ALPHA, sigma=SIGMA, gaussian_size=GAUSSIAN_SIZE):
    # Calculate the derivatives
    Ix = cv.Sobel(mat, cv.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_REFLECT)
    Iy = cv.Sobel(mat, cv.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_REFLECT)

    # Calculate the products of derivatives
    if gaussian_size > 1:
        Ixx = gaussian_filter(np.square(Ix), sigma=sigma,
                              mode='reflect', radius=gaussian_size//2)
        Iyy = gaussian_filter(np.square(Iy), sigma=sigma,
                              mode='reflect', radius=gaussian_size//2)
        Ixy = gaussian_filter(np.multiply(Ix, Iy), sigma=sigma,
                              mode='reflect', radius=gaussian_size//2)
    else:
        Ixx = np.square(Ix)
        Iyy = np.square(Iy)
        Ixy = np.multiply(Ix, Iy)

    orientation = np.arctan2(Iy, Ix) * 180 / np.pi

    detM = (Ixx * Iyy) - (Ixy ** 2)
    traceM = Ixx + Iyy
    R = detM - alpha * (traceM ** 2)  # corner strength function, R

    if 0 < threshold < 1:
        threshold = threshold * np.max(R)
    localMaxima = (R == maximum_filter(R, size=7, mode='reflect'))
    localMaxima = localMaxima * (R > threshold)
    interest_points = np.argwhere(localMaxima)

    keypoints = []
    for (y, x) in interest_points:
        keypoints.append(cv.KeyPoint(x.astype(np.float32), y.astype(
            np.float32), 1, angle=orientation[y, x], response=R[y, x]))
    return keypoints


# 2. Feature description
def createDescriptor(image, keypoints, descriptor='all'):

    if descriptor == 'orb':
        orb = cv.ORB_create()
        kp, des = orb.compute(image, keypoints)
        return kp, des
    elif descriptor == 'orb_fast':
        orb_fast = cv.ORB_create(scoreType=cv.ORB_FAST_SCORE)
        kp_fast, des_fast = orb_fast.detectAndCompute(image, None)
        return kp_fast, des_fast
    elif descriptor == 'orb_harris':
        orb_harris_default = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE)
        kp_harris, des_harris = orb_harris_default.detectAndCompute(
            image, None)
        return kp_harris, des_harris
    elif descriptor == 'all':
        orb = cv.ORB_create()
        kp, des = orb.compute(image, keypoints)

        orb_fast = cv.ORB_create(scoreType=cv.ORB_FAST_SCORE)
        kp_fast, des_fast = orb_fast.detectAndCompute(image, None)

        orb_harris_default = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE)
        kp_harris, des_harris = orb_harris_default.detectAndCompute(
            image, None)
        return {'orb': (kp, des), 'orb_fast': (kp_fast, des_fast), 'orb_harris': (kp_harris, des_harris)}


# 3. Feature matching
def SSDFeatureMatcher(des1, des2, limit=Config.LIMIT):
    matches = []
    distances = cdist(des1, des2, metric='sqeuclidean')
    for idx1, distance in enumerate(distances):
        idx2 = np.argmin(distance)
        matches.append(cv.DMatch(idx1, idx2, distance[idx2]))
    matches = sorted(matches, key=lambda x: x.distance)
    if limit >= len(matches):
        return matches
    return matches[:limit]

def RatioFeatureMatcher(des1, des2, limit=Config.LIMIT, ratio=Config.RATIO):
    matches = []
    distances = cdist(des1, des2, metric='sqeuclidean')
    for idx1, distance in enumerate(distances):
        idx2 = np.argmin(distance)
        d1 = distance[idx2]
        distance[idx2] = np.inf
        
        idx3 = np.argmin(distance)
        d2 = distance[idx3]
        
        if d1 / d2 < ratio:
            matches.append(cv.DMatch(idx1, idx2, d1))
    matches = sorted(matches, key=lambda x: x.distance)
    if limit >= len(matches):
        return matches
    return matches[:limit]    