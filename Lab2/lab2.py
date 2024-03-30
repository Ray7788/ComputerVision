import sys
import cv2 as cv
from scipy.ndimage import sobel, gaussian_filter, convolve, maximum_filter
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

SIGMA = 0.5
GAUSSIAN_SIZE = 5
ALPHA = 0.05
THRESHOLD = 1e7
RATIO = 0.7
LIMIT = 50


# 1. Feature detection
def HarrisPointsDetector(mat, threshold=THRESHOLD, alpha=ALPHA, sigma=SIGMA, gaussian_size=GAUSSIAN_SIZE):
    # Calculate the derivatives of x and y
    Ix = cv.Sobel(mat, cv.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_REFLECT)
    Iy = cv.Sobel(mat, cv.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_REFLECT)

    # Calculate the products of derivatives
    Ixx = gaussian_filter(np.square(Ix), sigma=sigma,
                          mode='reflect', radius=gaussian_size//2)
    Iyy = gaussian_filter(np.square(Iy), sigma=sigma,
                          mode='reflect', radius=gaussian_size//2)
    Ixy = gaussian_filter(np.multiply(Ix, Iy), sigma=sigma,
                          mode='reflect', radius=gaussian_size//2)

    # orientation of the gradient, in degrees
    orientation = np.arctan2(Iy, Ix) * 180 / np.pi

    detM = (Ixx * Iyy) - (Ixy ** 2)
    traceM = Ixx + Iyy
    R = detM - alpha * (traceM ** 2)  # corner strength function, R

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
def SSDFeatureMatcher(des1, des2, limit=LIMIT):
    matches = []
    distances = cdist(des1, des2, metric='sqeuclidean')
    for idx1, distance in enumerate(distances):
        idx2 = np.argmin(distance)
        matches.append(cv.DMatch(idx1, idx2, distance[idx2]))
    matches = sorted(matches, key=lambda x: x.distance)
    if limit >= len(matches):
        return matches
    return matches[:limit]

def RatioFeatureMatcher(des1, des2, limit=LIMIT, ratio=RATIO):
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

# Helper functions---------------------------------------------------------
def displayImages(images: list, labels: list, title: str):
    """
    Display multiple images in a single window
    """
    try:
        # Resize images to a smaller size
        # resized_images = [cv.resize(img, (200, 200)) for img in images]

        # Combine all images into a single image
        combined_image = np.hstack(images)

        # Display the combined image
        cv.imshow(title, combined_image)
        plt.show()
        cv.waitKey(0)
        cv.destroyAllWindows()
    except KeyboardInterrupt:
        cv.destroyAllWindows()
        sys.exit()


def drawKpImage(image, keypoints, color=(0, 255, 255), scale=1):
    keypoints_image = image.copy()
    keypoints_image = cv.drawKeypoints(
        image, keypoints, keypoints_image, color=color)
    keypoints_image = cv.resize(keypoints_image, (0, 0), fx=scale, fy=scale)
    return keypoints_image


def labelImage(image, text):
    image = image.copy()
    image = cv.putText(image, text, (40, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv.LINE_AA)
    return image


# Source data
bernie = cv.imread('bernieSanders.jpg', cv.IMREAD_COLOR)
bernie_dark = cv.imread('group/darkerBernie.jpg', cv.IMREAD_GRAYSCALE)
bernie_bright = cv.imread('group/brighterBernie.jpg', cv.IMREAD_GRAYSCALE)
bernie_blur = cv.imread('group/BernieMoreblurred.jpg', cv.IMREAD_GRAYSCALE)
bernie_180 = cv.imread('group/bernie180.jpg', cv.IMREAD_GRAYSCALE)
bernie_pixel = cv.imread('group/berniePixelated2.png', cv.IMREAD_GRAYSCALE)
bernie_noisy = cv.imread('group/bernieNoisy2.png', cv.IMREAD_GRAYSCALE)
# with something else
bernie_friends = cv.imread('group/bernieFriends.png', cv.IMREAD_GRAYSCALE)
bernie_salon = cv.imread(
    'group/bernieBenefitBeautySalon.jpeg', cv.IMREAD_GRAYSCALE)
bernie_school = cv.imread('group/bernieShoolLunch.jpeg', cv.IMREAD_GRAYSCALE)

# bernie_gray = cv.cvtColor(bernie, cv.COLOR_BGR2GRAY)
bernie_keypoints = HarrisPointsDetector(bernie)
bernie_descriptors = createDescriptor(bernie, bernie_keypoints)


def testHarrisDetector():
    """
    Test the Harris detector on different images
    """
    kp = drawKpImage(bernie, bernie_descriptors['orb'][0])
    kp_fast = drawKpImage(bernie, bernie_descriptors['orb_fast'][0])
    kp_harris = drawKpImage(bernie, bernie_descriptors['orb_harris'][0])

    try:
            cv.imwrite('output_group/bernie_kp.jpg', kp)
            cv.imwrite('output_group/bernie_kp_fast.jpg', kp_fast)
            cv.imwrite('output_group/bernie_kp_harris.jpg', kp_harris)
            print(' Saved images successfully!')
    except Exception as e:
            print("An error occurred while writing the images: ", e)
    displayImages([kp, kp_fast, kp_harris], [
                  'ORB', 'ORB_FAST', 'ORB_HARRIS'], 'Interest Points')


def testFeatureMatcher():
    """
    Test one image with different feature detectors
    """
    matches1 = SSDFeatureMatcher(
        bernie_descriptors['orb'][1], bernie_descriptors['orb_fast'][1], limit=20)
    matches2 = SSDFeatureMatcher(
        bernie_descriptors['orb'][1], bernie_descriptors['orb_harris'][1], limit=20)

    bernie_match_fast = cv.drawMatches(labelImage(bernie, 'self-implemented Harris'), bernie_descriptors['orb'][0], labelImage(
        bernie, 'OpenCV FAST'), bernie_descriptors['orb_fast'][0], matches1, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match_harris = cv.drawMatches(labelImage(bernie, 'self-implemented Harris'), bernie_descriptors['orb'][0], labelImage(
        bernie, 'OpenCV Harris'), bernie_descriptors['orb_harris'][0], matches2, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imshow('bernie_match_fast', cv.resize(
        bernie_match_fast, (0, 0), fx=0.25, fy=0.25))
    cv.imwrite('out_group/bernie_match_fast.jpg', bernie_match_fast)
    cv.imshow('bernie_match_harris', cv.resize(
        bernie_match_harris, (0, 0), fx=0.25, fy=0.25))
    cv.imwrite('out_group/bernie_match_harris.jpg', bernie_match_harris)


def testFeatureMatcher4(image):
    kp1, des1 = createDescriptor(image, HarrisPointsDetector(
        image, threshold=1e3), descriptor='orb')
    print(des1.shape)
    kp2, des2 = createDescriptor(image, HarrisPointsDetector(
        image, threshold=1e7), descriptor='orb')

    matches1 = SSDFeatureMatcher(bernie_descriptors['orb'][1], des1)
    matches2 = SSDFeatureMatcher(bernie_descriptors['orb'][1], des2)

    bernie_match1 = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(
        image, 'Dark'), kp1, matches1, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match2 = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(
        image, 'Dark'), kp2, matches2, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imwrite('out_group/bernie_match_th1e3.jpg', bernie_match1)
    cv.imwrite('out_group/bernie_match_th1e7.jpg', bernie_match2)

    cv.imshow('bernie_match_th1e3', cv.resize(
        bernie_match1, (0, 0), fx=0.25, fy=0.25))
    cv.imshow('bernie_match_th1e7', cv.resize(
        bernie_match2, (0, 0), fx=0.25, fy=0.25))


if __name__ == "__main__":
    # testHarrisDetector()
    testFeatureMatcher()
    # testFeatureMatcher4(darkerbernie)
    # testFeatureMatcher()

