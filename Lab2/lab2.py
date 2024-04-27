import os
import sys
import cv2 as cv
from scipy.ndimage import sobel, gaussian_filter, convolve, maximum_filter
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

SIGMA = 0.5
GAUSSIAN_SIZE = 5
ALPHA = 0.05
THRESHOLD = 18000   # best threshold for bernieSanders.jpg
# THRESHOLD = 50

RATIO = 0.7
LIMIT = 50
DIRECTORY = 'output_group'

if not os.path.exists(DIRECTORY):
    os.makedirs('output_group')

# 1. Feature detection
def HarrisPointsDetector(mat, gaussian_size=GAUSSIAN_SIZE, threshold=THRESHOLD, alpha=ALPHA, sigma=SIGMA):
    """
    Detect the Harris corner points in the given image
    """
    # Apply Gaussian filter to the image
    mat = cv.GaussianBlur(mat, (0, 0), 3)
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
    
    # Find local maxima in the corner strength matrix
    localMaxima = (R == maximum_filter(R, size=7, mode='reflect'))
    localMaxima = localMaxima * (R > threshold)
    interest_points = np.argwhere(localMaxima)

    keypoints = []
    for (y, x) in interest_points:
        keypoints.append(cv.KeyPoint(x.astype(np.float32), y.astype(
            np.float32), 1, angle=orientation[y, x], response=R[y, x]))
    return keypoints


# 2. Feature description
def featureDescriptor(image, keypoints, descriptor='orb'):
    """
    Create a descriptor for the given image and keypoints
    """
    if descriptor == 'orb': # ORB with self-designed Harris score
        orb = cv.ORB_create()
        kp, des = orb.compute(image, keypoints)
        return kp, des
    elif descriptor == 'orb_fast':  # ORB with FAST score
        orb_fast = cv.ORB_create(scoreType=cv.ORB_FAST_SCORE)
        kp_fast, des_fast = orb_fast.detectAndCompute(image, None)
        return kp_fast, des_fast
    elif descriptor == 'orb_harris':    # ORB with HARRIS score
        orb_harris_default = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE)
        kp_harris, des_harris = orb_harris_default.detectAndCompute(
            image, None)
        return kp_harris, des_harris



# 3. Feature matching
# Sum of squared difference
def SSDFeatureMatcher(des1, des2, limit=LIMIT):
    """
    Perform the SSD feature matching
    Returns the best matches(a list of DMatch objects)
    """
    best_matches = []
    # squared Euclidean distance
    distances = cdist(des1, des2, metric='sqeuclidean')
    for queryIdx, distance in enumerate(distances):
        trainIdx = np.argmin(distance)
        best_matches.append(cv.DMatch(queryIdx, trainIdx, distance[trainIdx]))
    best_matches = sorted(best_matches, key=lambda x: x.distance)
    if limit >= len(best_matches):
        return best_matches
    return best_matches[:limit]

# Ratio test
def RatioFeatureMatcher(des1, des2, limit=LIMIT, ratio=RATIO):
    """
    Perform the ratio test for feature matching
    """
    best_matches = []
    distances = cdist(des1, des2, metric='sqeuclidean')
    for queryIdx, distance in enumerate(distances):
        trainIdx = np.argmin(distance)
        distance1 = distance[trainIdx]
        distance[trainIdx] = np.inf
        # Find the second closest match to the query descriptor
        second_idx = np.argmin(distance)
        distance2 = distance[second_idx]
        
        if distance1 / distance2 < ratio:
            best_matches.append(cv.DMatch(queryIdx, trainIdx, distance1))
    best_matches = sorted(best_matches, key=lambda x: x.distance)
    if limit >= len(best_matches):
        return best_matches
    
    return best_matches[:limit]

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


def drawKpImage(image, keypoints, color=(0, 255, 0), scale=1):
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


# Source data input
bernie = cv.imread('group/bernieSanders.jpg', cv.IMREAD_GRAYSCALE)
# with different types of variations
bernie_dark = cv.imread('group/darkerBernie.jpg', cv.IMREAD_GRAYSCALE)  # with less brightness 
bernie_bright = cv.imread('group/brighterBernie.jpg', cv.IMREAD_GRAYSCALE) # with more brightness
bernie_blur = cv.imread('group/BernieMoreblurred.jpg', cv.IMREAD_GRAYSCALE) # with more blur
bernie_180 = cv.imread('group/bernie180.jpg', cv.IMREAD_GRAYSCALE)  # rotated 180 degrees
bernie_pixel = cv.imread('group/berniePixelated2.png', cv.IMREAD_GRAYSCALE)
bernie_noisy = cv.imread('group/bernieNoisy2.png', cv.IMREAD_GRAYSCALE)
# with something else
bernie_friends = cv.imread('group/bernieFriends.png', cv.IMREAD_GRAYSCALE)
bernie_salon = cv.imread(
    'group/bernieBenefitBeautySalon.jpeg', cv.IMREAD_GRAYSCALE)
bernie_school = cv.imread('group/bernieShoolLunch.jpeg', cv.IMREAD_GRAYSCALE)

# bernie_gray = cv.cvtColor(bernie, cv.COLOR_BGR2GRAY)
# Use bernieSanders.jpg as the original image and HarrisPointsDetector and default descriptor
bernie_keypoints = HarrisPointsDetector(bernie)
bernie_descriptors = ORBDescriptor(bernie, bernie_keypoints)


def testHarrisDetector():
    """
    Test the original image on the self-implemented Harris/ OpenCV's Harris/Fast detector
    The output should be some keypoints detected in the image
    """
    kp = drawKpImage(bernie, bernie_descriptors['orb'][0])
    kp_fast = drawKpImage(bernie, bernie_descriptors['orb_fast'][0])
    kp_harris = drawKpImage(bernie, bernie_descriptors['orb_harris'][0])

    try:
        cv.imwrite('output_group/bernie_kp_self.jpg', kp)
        cv.imwrite('output_group/bernie_kp_fast.jpg', kp_fast)
        cv.imwrite('output_group/bernie_kp_harris.jpg', kp_harris)
        print('Saved images successfully!')
    except Exception as e:
        print("An error occurred while writing the images: ", e)
    displayImages([kp, kp_fast, kp_harris], [
                  'ORB', 'ORB_FAST', 'ORB_HARRIS'], 'Interest Points')
    print("Done")

def testFeatureMatcher():
    """
    Test the original image with different feature detectors: ORB_FAST, ORB_HARRIS using the RatioFeatureMatcher
    output matcher images for each detector
    """
    matches1 = RatioFeatureMatcher(bernie_descriptors['orb'][1], bernie_descriptors['orb_fast'][1], limit=20)
    matches2 = RatioFeatureMatcher(bernie_descriptors['orb'][1], bernie_descriptors['orb_harris'][1], limit=20)

    bernie_match_fast = cv.drawMatches(labelImage(bernie, 'self-designed Harris'), bernie_descriptors['orb'][0], labelImage(bernie, 'OpenCV FAST'), bernie_descriptors['orb_fast'][0], matches1, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match_harris = cv.drawMatches(labelImage(bernie, 'self-designed Harris'), bernie_descriptors['orb'][0], labelImage(bernie, 'OpenCV Harris'), bernie_descriptors['orb_harris'][0], matches2, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imshow('bernie_match_fast', cv.resize(bernie_match_fast, (0,0), fx=0.25, fy=0.25))
    cv.imwrite('output_group/bernie_match_fast.jpg', bernie_match_fast)
    cv.imshow('bernie_match_harris', cv.resize(bernie_match_harris, (0,0), fx=0.25, fy=0.25))
    cv.imwrite('output_group/bernie_match_harris.jpg', bernie_match_harris)
    print("Done")

def testSingleImageFeatureMatcher(image, image_name='Dark'):
    """
    One certain image with different feature detectors: ORB, ORB_FAST, ORB_HARRIS using the RatioFeatureMatcher
    output matcher images for each detector
    suggest use images except the original image: bernie 
    """
    kp_orb, des_orb = ORBDescriptor(image, HarrisPointsDetector(image), descriptor='orb')
    kp_orb_fast, des_orb_fast = ORBDescriptor(image, HarrisPointsDetector(image), descriptor='orb_fast')
    kp_orb_harris, des_orb_harris = ORBDescriptor(image, HarrisPointsDetector(image), descriptor='orb_harris')

    matches_orb = RatioFeatureMatcher(bernie_descriptors['orb'][1], des_orb)
    matches_orb_fast = RatioFeatureMatcher(bernie_descriptors['orb_fast'][1], des_orb_fast)
    matches_orb_harris = RatioFeatureMatcher(bernie_descriptors['orb_harris'][1], des_orb_harris)

    bernie_match_orb = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(image, image_name), kp_orb, matches_orb, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match_orb_fast = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb_fast'][0], labelImage(image, image_name), kp_orb_fast, matches_orb_fast, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match_orb_harris = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb_harris'][0], labelImage(image, image_name), kp_orb_harris, matches_orb_harris, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imwrite(f'output_group/{image_name}_match_orb.jpg', bernie_match_orb)
    cv.imwrite(f'output_group/{image_name}_match_orb_fast.jpg', bernie_match_orb_fast)
    cv.imwrite(f'output_group/{image_name}_match_orb_harris.jpg', bernie_match_orb_harris)

    # displayImages([bernie_match_orb, bernie_match_orb_fast, bernie_match_orb_harris], ['ORB', 'ORB_FAST', 'ORB_HARRIS'], 'Interest Points')
    print("Done")

def testTwoFeatureMatchers(image):
    """
    Test the feature matcher on different matcher:
    SSDFeatureMatcher and RatioFeatureMatcher using the ORB descriptor
    suggest use darkBernie image
    """
    kp, des = ORBDescriptor(image, HarrisPointsDetector(image), descriptor='orb')
    ssd_matches = SSDFeatureMatcher(bernie_descriptors['orb'][1], des)
    ratio_matches = RatioFeatureMatcher(bernie_descriptors['orb'][1], des)
    
    bernie_match_ssd = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(image, 'Dark'), kp, ssd_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match_ratio = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(image, 'Dark'), kp, ratio_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv.imwrite('output_group/bernie_match_ssd.jpg', bernie_match_ssd)
    cv.imwrite('output_group/bernie_match_ratio.jpg', bernie_match_ratio)
    
    cv.imshow('bernie_match_ssd', cv.resize(bernie_match_ssd, (0,0), fx=0.25, fy=0.25))
    cv.imshow('bernie_match_ratio', cv.resize(bernie_match_ratio, (0,0), fx=0.25, fy=0.25))
    print("Done")


def testThresholdMatcher(image):
    """
    Test different threshold values using the SSD feature matcher
    """
    kp1, des1 = ORBDescriptor(image, HarrisPointsDetector(
        image, threshold=1e3), descriptor='orb')
    
    print(des1.shape)
    
    kp2, des2 = ORBDescriptor(image, HarrisPointsDetector(
        image, threshold=1e7), descriptor='orb')

    matches1 = SSDFeatureMatcher(bernie_descriptors['orb'][1], des1)
    matches2 = SSDFeatureMatcher(bernie_descriptors['orb'][1], des2)

    bernie_match1 = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(
        image, 'Dark'), kp1, matches1, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match2 = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(
        image, 'Dark'), kp2, matches2, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imwrite('output_group/bernie_match_th1e3.jpg', bernie_match1)
    cv.imwrite('output_group/bernie_match_th1e7.jpg', bernie_match2)

    cv.imshow('bernie_match_th1e3', cv.resize(
        bernie_match1, (0, 0), fx=0.25, fy=0.25))
    cv.imshow('bernie_match_th1e7', cv.resize(
        bernie_match2, (0, 0), fx=0.25, fy=0.25))
    print("Done")


def TestDifferentRatio(image, image_name='pixelated'):
    """
    Test the different ratio for RatioFeatureMatcher using the ORB descriptor
    """
    kp, des = ORBDescriptor(image, HarrisPointsDetector(image), descriptor='orb')
    
    matches_07 = RatioFeatureMatcher(bernie_descriptors['orb'][1], des, ratio=0.7)
    matches_05 = RatioFeatureMatcher(bernie_descriptors['orb'][1], des, ratio=0.5)
    matches_03 = RatioFeatureMatcher(bernie_descriptors['orb'][1], des, ratio=0.3)
    
    bernie_match_1 = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(image, image_name), kp, matches_07, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match_2 = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(image, image_name), kp, matches_05, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bernie_match_3 = cv.drawMatches(labelImage(bernie, 'Original'), bernie_descriptors['orb'][0], labelImage(image, image_name), kp, matches_03, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv.imwrite(f'output_group/{image_name}_bernie_match_07.jpg', bernie_match_1)
    cv.imwrite(f'output_group/{image_name}_bernie_match_05.jpg', bernie_match_2)
    cv.imwrite(f'output_group/{image_name}_bernie_match_03.jpg', bernie_match_3)

    cv.imshow('bernie_match_07', cv.resize(bernie_match_1, (0,0), fx=0.25, fy=0.25))
    cv.imshow('bernie_match_05', cv.resize(bernie_match_2, (0,0), fx=0.25, fy=0.25))
    cv.imshow('bernie_match_03', cv.resize(bernie_match_3, (0,0), fx=0.25, fy=0.25))
    
    displayImages([bernie_match_1, bernie_match_2, bernie_match_3], [f'ratio={r}' for r in [0.7, 0.5, 0.3]], 'Interest Points on different ratios')
    print("Done")


def testThresholdsDetector(thr_test_values):
    """
    Test different threshold values using the HarrisPointsDetector
    """
    bernie_thresholds = []
    detected_points_counts = []

    for i, thr in enumerate(thr_test_values):
        keypoints = HarrisPointsDetector(bernie, threshold=thr)
        detected_points_counts.append(len(keypoints))
        print(f"Number of keypoints detected: {len(keypoints)}")
        image = drawKpImage(bernie, keypoints)
        bernie_thresholds.append(image)
        filename = 'output_group/bernie_threshold_{:.2e}.jpg'.format(thr)
        cv.imwrite(filename, image)
        
    labels = ["threshold={:e}".format(thr) for thr in thr_test_values]
    displayImages(bernie_thresholds[:3], labels[:3], 'Interest Points with different thresholds')
    displayImages(bernie_thresholds[3:], labels[3:], 'Interest Points with different thresholds')

    # Plot the number of detected feature points vs. threshold values
    plt.plot(thr_test_values, detected_points_counts)
    plt.xlabel('Threshold Values')
    plt.ylabel('Number of Detected Feature Points')
    plt.title('Number of Detected Feature Points vs. Threshold Values')
    plt.grid(True)
    plt.savefig('feature_points_vs_threshold.png')

# Used for testing the feature matcher on different images --------------------------------
def getMatchImage(image, descriptor='orb'):
    """
    Get the match image for the given image and descriptor using the RatioFeatureMatcher
    """
    kp, des = ORBDescriptor(image, HarrisPointsDetector(image), descriptor='orb_fast')
    best_matches = RatioFeatureMatcher(bernie_descriptors[descriptor][1], des, limit=20)
    match_image = cv.drawMatches(bernie, bernie_descriptors[descriptor][0], image, kp, best_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_image

def testBernieMatches(images, labels):
    """
    Test the feature matcher on different images using the ORB descriptor
    """
    match_imgs = []
    for idx, img in enumerate(images):
        match_imgs.append(getMatchImage(img))
        cv.imshow(labels[idx], cv.resize(match_imgs[-1], (0,0), fx=0.5, fy=0.5))
        cv.imwrite(f'output_group/bernie_match_{labels[idx]}.jpg', match_imgs[-1])
    displayImages(match_imgs, labels, 'Bernie Sanders Matches')

if __name__ == "__main__":
    # testHarrisDetector()
    # testFeatureMatcher()
    # testTwoFeatureMatchers(bernie_dark) # Not very straightforward
    # testTwoFeatureMatchers(bernie_pixel)
    # testSingleImageFeatureMatcher(bernie_180, 'Roated 180 degrees') 
    # testThresholdMatcher(bernie_blur)--------
    # TestDifferentRatio(bernie_blur, 'Dark')

    # testBernieMatches([bernie_friends, bernie_salon, bernie_school], ['Friends', 'Salon', 'School'])
    # testBernieMatches([bernie_dark, bernie_bright], ['Dark', 'Bright'])
    # -------- testBernieMatches([bernie_180, bernie_pixel, bernie_noisy, bernie_blur], ['180', 'Pixel', 'Noisy', 'Blur'])

    # threshold_values = [1e7, 5e7, 1e8, 5e8, 1e9]  
    # Example threshold values to try [-1e3, 0, 1e3, 1e6, 1e7, 1e8]
    # testThresholdsDetector([1e2 ,5e2, 1e3, 5e3, 1e4, 5e4])

    print("End of the program")

