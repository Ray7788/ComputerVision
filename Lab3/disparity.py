import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


# Hyperparameters ================================
FOCAL_LENGTH_PIXELS = 5806.559
# For Focal Length Calculation
SENSOR_LENGTH_MM = 22.2
SENSOR_HEIGHT_MM = 14.8
PHOTO_LENGTH_PIXELS = 3088
PHOTO_HEIGHT_PIXELS = 2056
# Resize image
IMAGE_LENGTH_PIXELS = 740
IMAGE_HEIGHT_PIXELS = 505
# Depth into the scene
DOFFS = 114.291 # in pixels
BASELINE = 174.019 # in mm

# Set threshold values for
THR1 = 20
THR2 = 125
NUM_DISPARITY = 4 # *16 -----4
BLOCK_SIZE = 1 # *2 + 5 -----0

USE_EDGE_DETECTION = True

WINDOW_DIS = 'Disparity'
WINDOW_EDGE = 'Edge'


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
def calcRealFocalLength(sensor_size, image_size, focal_length):
    """
    Calculate the real focal length of the camera

    The formula is: f = (sensor_size / resolution) * focal_length
    sensor_size: the size of the camera sensor in mm
    image_size: the size of the image in pixels
    focal_length: the focal length of the camera in pixels
    """
    return (sensor_size / image_size) * focal_length

def getEdgeMap(img, threshold1, threshold2):
    """
    Get the edge map of the image using Canny edge detection

    img: the image to detect edges in
    threshold1: the first threshold for the hysteresis procedure
    threshold2: the second threshold for the hysteresis procedure
    """
    imgCanny = cv2.Canny(img, threshold1, threshold2)
    return imgCanny
# ================================================
#  1.3 Views of the Scene
def calcDepthMap(disparityMap, f=FOCAL_LENGTH_PIXELS, DOFFS=DOFFS, Z=BASELINE):
    """
    Calculate the depth map of the scene

    disparityMap: the disparity map of the scene

    baseLine: the distance between the two cameras in mm
    f: the focal length of the camera in pixels
    d: the disparity map of the scene in pixels
    Z: the depth map of the scene in mm

    The formula is: Z = baseLine * (f / (d + DOFFS))
    """

    Z = BASELINE * (f / (disparityMap + DOFFS))
    return Z

# ================================================
#
def plot(disparity):
    """
    Plot the 3D scene of the depth map
    """
    
    height, width = disparity.shape
    cx, cy = 0, 0
    x = []
    y = []
    z = []
    for r in range(height):
        for c in range(width):
                dis = disparity[r, c]
                if dis <= 1:
                    continue
                
                Z = calcDepthMap(dis, f=FOCAL_LENGTH_PIXELS, DOFFS=DOFFS, Z=BASELINE)
                X = (c - cx) * Z / FOCAL_LENGTH_PIXELS
                Y = (r - cy) * Z / FOCAL_LENGTH_PIXELS
                x.append(X)
                y.append(Y)
                z.append(Z)

    # Plt depths 3D
    ax = plt.axes(projection ='3d')
    ax.view_init(elev=30, azim=45)
    # Plot the 3D scatter plot
    ax.scatter(x, z, y, 'green', s=0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.savefig('3D.png', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()

    # elev=9, azim=-90 Top view
    ax = plt.axes(projection ='3d')
    ax.view_init(elev=90, azim=-90)
    ax.scatter(x, z, y, 'green', s=0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.savefig('myplot_90_-90.png', bbox_inches='tight')
    plt.show()

    # elev=20, azim=-120
    ax = plt.axes(projection ='3d')
    ax.view_init(elev=20, azim=-120)
    ax.scatter(x, z, y, 'green', s=0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.savefig('myplot_20_-120.png', bbox_inches='tight')
    plt.show()

    # elev=180, azim=0 side view
    ax = plt.axes(projection ='3d')
    ax.view_init(elev=180, azim=0)
    ax.scatter(x, z, y, 'green', s=0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.savefig('myplot_180_0.png', bbox_inches='tight')
    plt.show()

# ================================================
#
def concatImgs(imgs: list, labels: list):
    labeledImgs = []
    for i, img in enumerate(imgs):
        img = cv2.putText(img, labels[i], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        labeledImgs.append(img)
    concat = np.concatenate(labeledImgs, axis=1) 
    concat = cv2.resize(concat, (0,0), fx=0.5, fy=0.5)
    return concat

def drawDisparityImage(numDisparity = None, blockSize = None, threshold1=None, threshold2=None):
    
    if numDisparity is None:
        numDisparity = cv2.getTrackbarPos('Disparity:', WINDOW_DIS)
    if blockSize is None:
        blockSize = cv2.getTrackbarPos('Block Size:', WINDOW_DIS)
        
    if USE_EDGE_DETECTION:
        if threshold1 is None:
            threshold1 = cv2.getTrackbarPos('Threshold 1', WINDOW_EDGE)
        if threshold2 is None:
            threshold2 = cv2.getTrackbarPos('Threshold 2', WINDOW_EDGE)
            
        edgeL = getEdgeMap(imgL, threshold1, threshold2)
        edgeR = getEdgeMap(imgR, threshold1, threshold2)
        
        cv2.imshow(WINDOW_EDGE, edgeL)
    
        disparity = getDisparityMap(edgeL, edgeR, numDisparity*16, blockSize*2+5)
    else:
        disparity = getDisparityMap(imgL, imgR, numDisparity*16, blockSize*2+5)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow(WINDOW_DIS, disparityImg)
    
def drawEdgeDetected_threshold1(threshold1):
    drawDisparityImage(threshold1=threshold1)
    
def drawEdgeDetected_threshold2(threshold2):
    drawDisparityImage(threshold2=threshold2)
    
def drawDisparity_numDisparity(numDisparity):
    drawDisparityImage(numDisparity=numDisparity)
    
def drawDisparity_blockSize(blockSize):
    drawDisparityImage(blockSize=blockSize)

def compute_and_display_disparity(imgL, imgR, USE_EDGE_DETECTION, THR1, THR2, NUM_DISPARITY, BLOCK_SIZE, WINDOW_EDGE, WINDOW_DIS):
    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    if USE_EDGE_DETECTION:
        cv2.imshow(WINDOW_EDGE, getEdgeMap(imgL, THR1, THR2))
        
        edgeL = getEdgeMap(imgL, THR1, THR2)
        edgeR = getEdgeMap(imgR, THR1, THR2)
        
        cv2.createTrackbar('Threshold 1', WINDOW_EDGE, THR1, 255, drawEdgeDetected_threshold1)
        cv2.createTrackbar('Threshold 2', WINDOW_EDGE, THR2, 255, drawEdgeDetected_threshold2)

        # Get disparity map
        disparity = getDisparityMap(edgeL, edgeR, NUM_DISPARITY*16, 5+BLOCK_SIZE*2)
    else:
        # Get disparity map
        disparity = getDisparityMap(imgL, imgR, NUM_DISPARITY*16, 5+BLOCK_SIZE*2)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.imwrite('disparity_no_canny.png', disparityImg*255)
    # Show result
    cv2.imshow(WINDOW_DIS, disparityImg)

    cv2.createTrackbar('Disparity:', WINDOW_DIS, NUM_DISPARITY, 16, drawDisparity_numDisparity)
    cv2.createTrackbar('Block Size:', WINDOW_DIS, BLOCK_SIZE, 16, drawDisparity_blockSize)

    cv2.imwrite('disparity.png', disparityImg*255)

    depth = calcDepthMap(disparity)
    # Show 3D plot of the scene
    plot(depth)


if __name__ == '__main__':
    # Calculate the real focal length of the camera
    focal_length = calcRealFocalLength(SENSOR_LENGTH_MM, PHOTO_LENGTH_PIXELS, FOCAL_LENGTH_PIXELS)

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #  Check if the image was loaded
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    print("Please wait while the disparity map is calculated...")
    # # 1.2  Disparity Map
    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    if USE_EDGE_DETECTION:
        cv2.imshow(WINDOW_EDGE, getEdgeMap(imgL, THR1, THR2))
        
        edgeL = getEdgeMap(imgL, THR1, THR2)
        edgeR = getEdgeMap(imgR, THR1, THR2)
        
        cv2.createTrackbar('Threshold 1', WINDOW_EDGE, THR1, 255, drawEdgeDetected_threshold1)
        cv2.createTrackbar('Threshold 2', WINDOW_EDGE, THR2, 255, drawEdgeDetected_threshold2)

        # Get disparity map
        disparity = getDisparityMap(edgeL, edgeR, NUM_DISPARITY*16, 5+BLOCK_SIZE*2)
    else:
        # Get disparity map
        disparity = getDisparityMap(imgL, imgR, NUM_DISPARITY*16, 5+BLOCK_SIZE*2)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.imwrite('disparity_no_canny.png', disparityImg*255)
    # Show result
    cv2.imshow(WINDOW_DIS, disparityImg)

    cv2.createTrackbar('Disparity:', WINDOW_DIS, NUM_DISPARITY, 16, drawDisparity_numDisparity)
    cv2.createTrackbar('Block Size:', WINDOW_DIS, BLOCK_SIZE, 16, drawDisparity_blockSize)

    cv2.imwrite('disparity.png', disparityImg*255)



    # compute_and_display_disparity(imgL, imgR, USE_EDGE_DETECTION, THR1, THR2, NUM_DISPARITY, BLOCK_SIZE, WINDOW_EDGE, WINDOW_DIS)

    # Compare different values of numDisparities and blockSize ------------------------------------ #
    # images = []
    # labels = []
    # for i in range(4):
    #     val = i * 16
    #     labels.append(f'numDisparities: {val}')
    #     disparity = getDisparityMap(edgeL, edgeR, val, 5+BLOCK_SIZE*2)
    #     disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    #     images.append(disparityImg)
    # output = concatImgs(images, labels)
    # cv2.imwrite('report/disparity_compare_numDisparities_1.png', output*255)
    # cv2.imshow('comparison', output)
    
    # # Compare different values of numDisparities and blockSize ------------------------------------ #
    # images = []
    # labels = []
    # for i in range(4):
    #     val = (4+i) * 16
    #     labels.append(f'numDisparities: {val}')
    #     disparity = getDisparityMap(edgeL, edgeR, val, 5+BLOCK_SIZE*2)
    #     disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    #     images.append(disparityImg)
    # output = concatImgs(images, labels)
    # cv2.imwrite('report/disparity_compare_numDisparities_2.png', output*255)
    # cv2.imshow('comparison', output)

    # # Compare different values of blockSize ------------------------------------------------------ #
    # images = []
    # labels = []    
    # for i in range(4):
    #     val = 5 + i*2
    #     labels.append(f'blockSize: {val}')
    #     disparity = getDisparityMap(edgeL, edgeR, NUM_DISPARITY*16, val)
    #     disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    #     images.append(disparityImg)
    # output = concatImgs(images, labels)
    # cv2.imwrite('report/disparity_compare_blockSize_1.png', output*255)
    # cv2.imshow('comparison', output)

    # # Compare different values of blockSize ------------------------------------------------------ #
    # images = []
    # labels = []    
    # for i in range(4):
    #     val = 5 + (4+i)*2
    #     labels.append(f'blockSize: {val}')
    #     disparity = getDisparityMap(edgeL, edgeR, NUM_DISPARITY*16, val)
    #     disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    #     images.append(disparityImg)
    # output = concatImgs(images, labels)
    # cv2.imwrite('report/disparity_compare_blockSize_2.png', output*255)
    # cv2.imshow('comparison', output)
    
    # Calculate the depth map of the scene -------------------------------------------------------- #
    # depth = calcDepthMap(disparity)

    # Show 3D plot of the scene
    plot(disparity)

    # Wait for user to press space or escape
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
    print('Done.')
    sys.exit()
