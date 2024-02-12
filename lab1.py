import numpy as np
import cv2


# Function for performing a convolution using a 3x3 kernel
def convolution(image, kernel):
    # Pad the image manually
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

    # Get dimensions of the image and the kernel
    rows, cols = image.shape

    # Define the new image with sizes without the padding
    conv_image = np.zeros((rows, cols), dtype=np.float32)

    # Loop over the image and compute the values of the new smoothed image
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            conv_image[i - 1, j - 1] = np.sum(padded_image[i - 1:i + 2, j - 1:j + 2] * kernel)

    # Normalize the convoluted image
    # conv_image /= np.sum(kernel)

    # # Display it on the screen and save the image
    # cv2.imshow("Convoluted Kitty", conv_image.astype(np.uint8))
    # cv2.imwrite("smoothKitty.jpg", conv_image)
    return conv_image

# Function for thresholding the edge strength image
def threshold_edge_strength(image, threshold):
    # Apply thresholding
    thresholded_image = np.where(image >= threshold, 255, 0).astype(np.uint8)
    return thresholded_image

def main():
    # Load the image
    image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)
    # Define the weighted kernel (e.g., Gaussian)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])

    # Perform convolution
    result = convolution(image, kernel)

    # Display the original and convoluted images
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define the horizontal and vertical Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Compute horizontal and vertical gradients
    gradient_x = convolution(image, sobel_x)
    gradient_y = convolution(image, sobel_y)
    edge_strength = np.sqrt(gradient_x**2 + gradient_y**2)

    # Thresholding the edge strength image
    threshold = 30  # Adjust threshold value as needed
    thresholded_image = threshold_edge_strength(edge_strength, threshold)


    # Display the original, horizontal gradient, vertical gradient, and edge strength images
    cv2.imshow('Original Image', image)
    cv2.imshow('Horizontal Gradient', gradient_x.astype(np.uint8))
    cv2.imshow('Vertical Gradient', gradient_y.astype(np.uint8))
    cv2.imshow('Edge Strength', edge_strength.astype(np.uint8))
    cv2.imshow('Thresholded Edge Strength', thresholded_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()