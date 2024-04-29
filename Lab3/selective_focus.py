import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# ================================================
FOCAL_LENGTH_PIXELS = 5806.559
SENSOR_LENGTH_MM = 22.2
SENSOR_HEIGHT_MM = 14.8
PHOTO_LENGTH_PIXELS = 3088
PHOTO_HEIGHT_PIXELS = 2056
DOFFS = 114.291 # in pixels
BASELINE = 174.019 # in mm

NUM_DISPARITY = 1 # *16
BLOCK_SIZE = 20 # *2 + 5    # Use a larger block size 

GAUSSIAN_SIZE = 10 # *2 + 1
FOREGROUND_THRESHOLD = 20

K = 3

USE_EDGE_DETECTION = False

WINDOW = 'Depth'


# ================================================
# 1.2 Disparity Map
def getDisparityMap(imL, imR, numDisparities, blockSize):
    """
    Get the disparity map of the scene

    imL: the left image
    imR: the right image
    numDisparities: the maximum disparity minus minimum disparity
    blockSize: the size of the window used to match pixels
    """
    # Ensure blockSize is odd, within 5..255, and not larger than image width or height
    # blockSize = max(5, min(255, blockSize))
    # if blockSize % 2 == 0:
    #     blockSize += 1
    # blockSize = min(blockSize, imL.shape[1], imL.shape[0])

    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================
# 1.1 Focal Length Calculation
def calcRealFocalLength(sensorSize, imgSize, focalLength):
    """
    Calculate the real focal length of the camera

    The formula is: f = (sensorSize / resolution) * focalLength
    sensorSize: the size of the camera sensor in mm
    imgSize: the size of the image in pixels
    focalLength: the focal length of the camera in pixels
    """
    return (sensorSize / imgSize) * focalLength

def getEdgeMap(img, threshold1, threshold2):
    """
    Get the edge map of the image using Canny edge detection

    img: the image to detect edges in
    threshold1: the first threshold for the hysteresis procedure
    threshold2: the second threshold for the hysteresis procedure
    """
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    imgCanny = cv2.Canny(img, threshold1, threshold2)
    return imgCanny
# ================================================
# 2 Selective Focus
def calcDepth(disparity, k):
    """
    Calculate the depth map from the disparity map
    """
    depth = 1 / (disparity + k)
    return depth

def backgroundBlur(img, depth, gaussian_size=25, depth_threshold=0.2):
    """
    Blur the background of the image based on the depth map

    img: the image to blur
    depth: the depth map of the scene
    gaussian_size: the size of the Gaussian kernel
    depth_threshold: the threshold for the depth map
    """
    foreground_depth = depth.min() + depth_threshold * (depth.max() - depth.min())
    depth_normalized = (depth - depth.min()) / (depth.max() - foreground_depth)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    blurred = cv2.GaussianBlur(img, (gaussian_size, gaussian_size), 0, borderType=cv2.BORDER_REFLECT).astype(np.float64)

    output = blurred * depth_normalized + img * (1 - depth_normalized)    
    return output.astype(np.uint8)


# ================================================
#
def concatImages(img1, img2, label1='img1', label2='img2'):
    height = max(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))
    
    img1 = cv2.putText(img1, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    img2 = cv2.putText(img2, label2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    output = np.concatenate((img1, img2), axis=1)
    return output

def plot(depth):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    low_threshold = 5660 #40
    high_threshold = 8800 #60
    
    f = calcRealFocalLength(SENSOR_LENGTH_MM, PHOTO_LENGTH_PIXELS, FOCAL_LENGTH_PIXELS)
    rows, cols = np.where((depth > low_threshold) & (depth < high_threshold))
    z = depth[(depth > low_threshold) & (depth < high_threshold)]
    # x = (cols / len(cols) * PHOTO_HEIGHT_PIXELS) * (SENSOR_LENGTH_MM / PHOTO_HEIGHT_PIXELS) * (z / f)
    # y = (SENSOR_HEIGHT_MM / rows) * (z / f)
    x = cols
    y = rows
    print('finished')
    
    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.view_init(elev=30, azim=45)
    ax.scatter(x, z, y, 'green', s=0.1)

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    plt.savefig('myplot.pdf', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()


# ================================================
#


def drawDepth(numDisparity = None, blockSize = None, k = None, gaussianSize = None, foregroundThreshold = None):
    if numDisparity is None:
        numDisparity = cv2.getTrackbarPos('Disparity', WINDOW)
    if blockSize is None:
        blockSize = cv2.getTrackbarPos('Block Size', WINDOW)
    if k is None:
        k = cv2.getTrackbarPos('K value', WINDOW)
    if gaussianSize is None:
        gaussianSize = cv2.getTrackbarPos('Gaussian Size', WINDOW)
    if foregroundThreshold is None:
        foregroundThreshold = cv2.getTrackbarPos('Foreground', WINDOW)
        
    gaussianSize = gaussianSize * 2 + 1
    
    disparity = getDisparityMap(imgL, imgR, numDisparities=numDisparity*16, blockSize=blockSize*2+5)
    depth = calcDepth(disparity, k)
    depth = np.interp(depth, (depth.min(), depth.max()), (0.0, 225.0))
    
    depth = cv2.GaussianBlur(depth, (gaussianSize, gaussianSize), 0, borderType=cv2.BORDER_DEFAULT).astype(np.uint8)
    
    blurred = backgroundBlur(imgL, depth, depth_threshold=foregroundThreshold/100)    
    
    concatImg = concatImages(blurred, depth, 'Blurred Image', 'Calculated Depth Map')
    
    cv2.imshow(WINDOW, concatImg)
    
    return concatImg

    
def drawDepth_numDisparity(numDisparity):
    drawDepth(numDisparity = numDisparity)
    
def drawDepth_blockSize(blockSize):
    drawDepth(blockSize = blockSize)
    
def drawDepth_k(k):
    drawDepth(k = k)
    
def drawDepth_gaussianSize(gaussianSize):
    drawDepth(gaussianSize = gaussianSize)
    
def drawDepth_foregroundThreshold(foregroundThreshold):
    drawDepth(foregroundThreshold = foregroundThreshold)

if __name__ == '__main__':

    focal_length = calcRealFocalLength(SENSOR_LENGTH_MM, PHOTO_LENGTH_PIXELS, FOCAL_LENGTH_PIXELS)

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgL_colour = cv2.imread('girlL.png')
    # Check if the image was successfully loaded
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename =  'girlR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgR_colour = cv2.imread('girlR.png')
    #   Check if the image was successfully loaded
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # ================================================
    # Create a window to display the image in
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    
    cv2.createTrackbar('Disparity', WINDOW, NUM_DISPARITY, 16, drawDepth_numDisparity)
    cv2.createTrackbar('Block Size', WINDOW, BLOCK_SIZE, 32, drawDepth_blockSize)
    cv2.createTrackbar('K value', WINDOW, K, 100, drawDepth_k)
    cv2.createTrackbar('Gaussian Size', WINDOW, GAUSSIAN_SIZE, 100, drawDepth_gaussianSize)
    cv2.createTrackbar('Foreground', WINDOW, FOREGROUND_THRESHOLD, 100, drawDepth_foregroundThreshold)

    output = drawDepth(numDisparity = NUM_DISPARITY, blockSize = BLOCK_SIZE, k = K, gaussianSize = GAUSSIAN_SIZE, foregroundThreshold = FOREGROUND_THRESHOLD)
    
    cv2.imwrite('report/focus_image.png', output)

    # Wait for a key press
    while True:
        key = cv2.waitKey(0)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
