import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys


def custom_normalize(image, alpha=0, beta=1, norm_type='minmax', dtype=np.float64):
    """
    Custom implementation of normalization similar to cv2.normalize function.

    Parameters:
        image: numpy.ndarray
            The input image.
        alpha: float, optional
            The lower bound of the normalization range.
        beta: float, optional
            The upper bound of the normalization range.
        norm_type: str, optional
            The type of normalization. Options are 'minmax' for min-max normalization
            and 'meanstd' for mean and standard deviation normalization.
        dtype: type, optional
            The data type of the output array.

    Returns:
        numpy.ndarray
            The normalized image."""
    if norm_type == 'minmax':
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / (max_val -
                                                min_val) * (beta - alpha) + alpha
    elif norm_type == 'meanstd':
        mean_val = np.mean(image)
        std_val = np.std(image)
        normalized_image = (image - mean_val) / std_val
        normalized_image = np.clip(normalized_image, alpha, beta)
    else:
        raise ValueError(
            "Invalid normalization type. Choose 'minmax' or 'meanstd'.")

    return normalized_image.astype(dtype)

# Example usage:
# Assuming 'padded' is your input image
# padded64 = custom_normalize(padded, alpha=0, beta=1, norm_type='minmax', dtype=np.float64)


def np_to_cv_type(x):
    """Returns an OpenCV type from a Numpy type.
    Args:
        x: Numpy type

    https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python

    Returns:
        OpenCV type
        Alpha
        Beta    
    """
    return {
        np.uint8: (cv.CV_8U, 0, 255),
        np.float32: (cv.CV_32F, 0, 1),
        np.float64: (cv.CV_64F, 0, 1)
    }.get(x, (cv.CV_8U, 0, 255))


def mean_distribution(row, col, ktype=np.float32):
    """Generates an average distribution mxn matrix, where all elements are 1/(row*col).
    Used for mean filtering."""
    return np.ones((row, col), dtype=ktype) / (row * col)

def gaussian_distribution(row, col, amp, sigma,
                            cx=0, cy=0,
                            ktype=np.float32,
                            normalize=True):
    """Generates a Gaussian distribution mxn matrix.
    Args:
        row: number of rows
        col: number of columns
        amp: amplitude
        sigma: standard deviation
        cx/cy: center on X/Y axis
        ktype: type of the matrix
        normalize: normalize"""
    kernel = np.empty((row, col), dtype=np.float64)

    # Compute anchor point
    ax, ay = row // 2, col // 2
    # Fill matrix with results from the 2D Gaussian function
    total = 0.0
    for i in range(0, row):
        for j in range(0, col):
            x = i - ax
            y = j - ay
            gx = ((x-cx)**2) / (2 * sigma**2)
            gy = ((y-cy)**2) / (2 * sigma**2)
            value = amp * np.exp(-(gx + gy)) / (2 * np.pi * sigma**2)
            kernel[i, j] = value
            total = total + value

    if normalize:    # Normalize numbers by default so they add up to 1
        kernel = kernel / total

    return kernel.astype(ktype)


def padding_matrix(m, v=0, width=1, height=1):
    """Pads a matrix with a given value for image convolution.
    Args:
        m: matrix
        v: value
        width: padding width
        height: padding height"""
    dx = m.shape[0] + 2 * width
    dy = m.shape[1] + 2 * height
    result = np.full((dx, dy), v, dtype=m.dtype)
    result[width:m.shape[0]+width, height:m.shape[1]+height] = m
    return result


def thresholding_binary(img, threshold):
    """Performs binary thresholding on an image.
    Args:
        img: input image
        threshold: threshold"""
    new_img = np.zeros(img.shape, np.uint8)
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if img[i,j] > threshold:
                new_img[i,j] = 255
            else:
                new_img[i,j] = 0
    return new_img


def convolution(m, k):
    """Performs 2D convolution on an image.
    Args:
        m: input image
        k: kernel"""
    kx, ky = k.shape[0], k.shape[1]
    assert kx % 2 == 1 and ky % 2 == 1, "convolution error: kernel dimensions must be odd numbers; got {} and {}".format(
        kx, ky)

    orig_type, alpha, beta = np_to_cv_type(m.dtype)    # Remember the original type
    px, py = kx // 2, ky // 2   # Compute what padding is required

    # Pad matrix to deal with edges and corners and convert it to float64
    padded = padding_matrix(m, 0, px, py)

    # padded64 = cv.normalize(padded, None, alpha=0, beta=1,
    #                         norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    padded64 = custom_normalize(
        padded, alpha=0, beta=1, norm_type='minmax', dtype=np.float64)

    # Initialize result matrix
    result64 = np.empty(m.shape, dtype=np.float64)

    # Loop through the image and perform the convolution
    nx, ny = m.shape
    for i in range(0, nx):
        for j in range(0, ny):
            region = padded64[i:i+kx, j:j+ky]
            result64[i, j] = np.sum(region * k)

    # Return part of the padded matrix that forms the output
    # result = cv.normalize(result64, None, alpha=0, beta=1,
    #                        norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    # Return part of the padded matrix that forms the output
    result = cv.normalize(np.absolute(result64), None, alpha=alpha, beta=beta,
                          norm_type=cv.NORM_MINMAX, dtype=orig_type)
    # result = custom_normalize(np.absolute(
    #     result64), alpha=alpha, beta=beta, norm_type='minmax', dtype=orig_type)
    return result


def mean_filter(img, krow, kcol, ktype=np.float32):
    """Performs mean filtering on the input image
    Args:
        img: input image
        krow: number of rows in kernel
        kcol: number of columns in kernel
        ktype: type"""
    kernel = mean_distribution(krow, kcol, ktype=ktype)
    return convolution(img, kernel)


def gaussian_filter(img, krow, kcol, amp, sigma, cx=0, cy=0,
                      ktype=np.float32, normalize=True):
    """
    Performs Gaussian filtering on the input image
    Args:
        img: input image
        krow: number of rows in kernel
        kcol: number of columns in kernel
        amp: amplitude
        sx: spread on X axis
        sy: spread on Y axis
        cx: center on X axis
        cy: center on Y axis
        ktype: type
        normalize: normalize"""
    kernel = gaussian_distribution(krow, kcol, amp, sigma, cx=cx, cy=cy,
                                     ktype=ktype, normalize=normalize)
    return convolution(img, kernel)


def calc_gradient(img, theresholding_val, 
                horizontal_gradient_kernel, vertical_gradient_kernel,
                sobelX_kernel, sobelY_kernel,
                filter, title=None,
                show_img=True, draw_hist=False, hist_title=""):
    """Calculates the gradient of an image using the Sobel operator.

    Args:
        img: input image
        theresholding_val: thresholding value
        horizontal_gradient_kernel: horizontal gradient kernel
        vertical_gradient_kernel: vertical gradient kernel
        sobelX_kernel: Sobel X kernel
        sobelY_kernel: Sobel Y kernel
        filter: filter type
        title: title
        show_img: show image
        draw_hist: draw histogram
        hist_title: histogram title

    Returns:
        img_blur: blurred image
        horizontal_gradient_img: horizontal gradient image
        vertical_gradient_img: vertical gradient image
        combined_gradient_img: combined gradient image
        theresholded_img: thresholded image
        hist: histogram"""

    if filter == "mean_filter":
        img_blur = mean_filter(img, horizontal_gradient_kernel, vertical_gradient_kernel, ktype=np.float64)
    else:
        img_blur = gaussian_filter(img, horizontal_gradient_kernel, vertical_gradient_kernel, 5, 0.25, 0.25, ktype=np.float64)
    
    horizontal_gradient_img = convolution(img_blur, sobelX_kernel)
    vertical_gradient_img = convolution(img_blur, sobelY_kernel)
    # combined_gradient_img = np.sqrt(np.square(horizontal_gradient_img, dtype=np.float32) + np.square(vertical_gradient_img, dtype=np.float32), dtype=np.float32)
    # combined_gradient_img = post_process(combined_gradient_img)
    combined_gradient_img = cv.addWeighted(horizontal_gradient_img, 0.5,
                                       vertical_gradient_img, 0.5, 0)

    # 2.3 Threshold to find edges
    theresholded_img = thresholding_binary(combined_gradient_img, theresholding_val)
        
    # Compute histogram
    hist = cv.calcHist([combined_gradient_img], [0], None, [256], [0, 256])
    hist = hist.reshape(256)
    if draw_hist:           # plot the histogram
        plt.plot(hist)
        plt.xlim([0, 256])
    
    # if show_img:
    #     combine_images([horizontal_gradient_img, vertical_gradient_img, combined_gradient_img, theresholded_img], title, labels=['horizontal', 'vertical', 'edge strength', f'thereshold({theresholding_val})'])
    #     cv.imshow('theresholded_img_'+title, theresholded_img)
    #     cv.createTrackbar('threshold_'+title, 'theresholded_img_'+title, theresholding_val, 255, set_threshold)
    # else:
    return img_blur, horizontal_gradient_img, vertical_gradient_img, combined_gradient_img, theresholded_img, hist


if __name__ == "__main__":
    print("OpenCV version " + cv.__version__)
    filepath = "kitty.bmp"
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)  # Open image in grayscale
    assert img is not None, "error: imread: failed to open " + filepath

    # Define the Sobel kernels https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
    # https://en.wikipedia.org/wiki/Sobel_operator
    sobelX_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float64)

    sobelY_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]], dtype=np.float64)

    # Define experiment parameters
    GK_SIZE = 7  # Gaussian kernel size
    GK_SIZE_2 = 3  # Gaussian kernel size
    GK_AMP = 5  # Gaussian kernel amplitude
    GK_SX = 0.25  # Gaussian kernel spread on X axis
    GK_SY = 0.25  # Gaussian kernel spread on Y axis
    THRESH_VALUE = 25
    WINDOW_NAME = 'COMP37212 Lab1'

    # Start comparison ----------------------------------------------------------
    ### 1st experiment Average Filterings V.S. Weighted Average (Gaussian) Filtering
    # 2. Perform experiment for a mean filter kernel    ***********************
    img_mean_blur, img_mean_sobelX, img_mean_sobelY, img_mean_gradient, img_mean_edges, img_mean_hist = calc_gradient(img, THRESH_VALUE, 
                                                                                                       GK_SIZE, GK_SIZE, 
                                                                                                       sobelX_kernel, sobelY_kernel,
                                                                                                       "mean_filter")
    # 4. Perform experiment for an weighted-mean filter kernel  ***********************
    img_gaussian_blur, img_gaussian_sobelX, img_gaussian_sobelY, img_gaussian_gradient, img_gaussian_edges, img_gaussian_hist = calc_gradient(img, THRESH_VALUE,
                                                                                                                            GK_SIZE, GK_SIZE,
                                                                                                                            sobelX_kernel, sobelY_kernel,
                                                                                                                            "gaussian_filter")
    # Compare edge strength images
    img_edge_comparison = img_mean_edges - img_gaussian_edges

    ### 2nd experiment for different kernel size --------------------------------
    # Use Gaussian kernel with different sizes
    img_gaussian_blur_, img_gaussian_sobelX_, img_gaussian_sobelY_, img_gaussian_gradient_, img_gaussian_edges_, _ = calc_gradient(img, THRESH_VALUE, 
                                                                                                       GK_SIZE_2, GK_SIZE_2, 
                                                                                                       sobelX_kernel, sobelY_kernel,
                                                                                                       "mean_filter", mean_filter)

    ### 3rd experiment for different thresholding --------------------------------


    # Show all images in a single window for easy control with a slider ------------
    # 1st row --------------------- #
    horizontal = np.concatenate((img_mean_blur, img_mean_sobelX), axis=1)
    horizontal = np.concatenate((horizontal, img_mean_sobelY), axis=1)
    horizontal = np.concatenate((horizontal, img_mean_gradient), axis=1)
    horizontal = np.concatenate((horizontal, img_mean_edges), axis=1)
    horizontal = np.concatenate((horizontal, img), axis=1)
    # 2nd row --------------------- #
    vertical = np.concatenate((img_gaussian_blur, img_gaussian_sobelX), axis=1)
    vertical = np.concatenate((vertical, img_gaussian_sobelY), axis=1)
    vertical = np.concatenate((vertical, img_gaussian_gradient), axis=1)
    vertical = np.concatenate((vertical, img_gaussian_edges), axis=1)
    vertical = np.concatenate((vertical, img_edge_comparison), axis=1)
    window = np.concatenate((horizontal, vertical), axis=0)
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    cv.imshow(WINDOW_NAME, window)

    # Display histograms for image thresholding
    mean_plot = plt.figure('Mean kernel image histogram')
    plt.bar(np.linspace(0, 255, 256), img_mean_hist)
    plt.title('Histogram')
    plt.title('Gray level')
    plt.ylabel('Frequency')
    gaussian_plot = plt.figure('Weighted-mean kernel image histogram')
    plt.bar(np.linspace(0, 255, 256), img_gaussian_hist)
    plt.title('Histogram')
    plt.title('Gray level')
    plt.ylabel('Frequency')
    plt.show()

    # Wait for key press from user to keep windows alive
    # If user presses 'S' key, the images are saved on disk
    k = cv.waitKey(0)
    if k == ord('s'):
        print('Saving img_mean_blur.jpg')
        cv.imwrite('img_mean_blur.jpg', img_mean_blur)
        print('Saving img_mean_sobelX.jpg')
        cv.imwrite('img_mean_sobelX.jpg', img_mean_sobelX)
        print('Saving img_mean_sobelY.jpg')
        cv.imwrite('img_mean_sobelY.jpg', img_mean_sobelY)
        print('Saving img_mean_gradient.jpg')
        cv.imwrite('img_mean_gradient.jpg', img_mean_gradient)
        print('Saving img_mean_hist.jpg')
        mean_plot.savefig('img_mean_hist.jpg')
        print('Saving img_mean_edges.jpg')
        cv.imwrite('img_mean_edges.jpg', img_mean_edges)
        print('Saving img_gaussian_blur.jpg')
        cv.imwrite('img_gaussian_blur.jpg', img_gaussian_blur)
        print('Saving img_gaussian_sobelX.jpg')
        cv.imwrite('img_gaussian_sobelX.jpg', img_gaussian_sobelX)
        print('Saving img_gaussian_sobelY.jpg')
        cv.imwrite('img_gaussian_sobelY.jpg', img_gaussian_sobelY)
        print('Saving img_gaussian_gradient.jpg')
        cv.imwrite('img_gaussian_gradient.jpg', img_gaussian_gradient)
        print('Saving img_gaussian_hist.jpg')
        mean_plot.savefig('img_gaussian_hist.jpg')
        print('Saving img_gaussian_edges.jpg')
        cv.imwrite('img_gaussian_edges.jpg', img_gaussian_edges)
        print('Saving img_edge_comparison.jpg')
        cv.imwrite('img_edge_comparison.jpg', img_edge_comparison)
        print('All images saved!')
    else:
        cv.destroyAllWindows()
    sys.exit(0)
