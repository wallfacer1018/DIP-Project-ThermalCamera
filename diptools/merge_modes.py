import cv2
import frequency
import regulator
import numpy as np

def merge_grayscale(img1, img2, ratio):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same size if they are not already
    # height, width = gray1.shape
    # gray2 = cv2.resize(gray2, (width, height))

    # Merge images with the given ratio
    merged_image = cv2.addWeighted(gray1, ratio, gray2, 1 - ratio, 0)

    return merged_image

def merge_highpass(img1, img2, ratio, sigma):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    H_p = frequency.H_gauss_LPF(gray1.shape, sigma, double_pad=True)
    H_p = 1 - H_p
    gray1 = frequency.filter_freq_pad(gray1, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same size if they are not already
    # height, width = gray1.shape
    # gray2_resized = cv2.resize(gray2, (width, height))

    # Merge images with the given ratio
    merged_image = cv2.addWeighted(gray1, ratio, gray2, 1 - ratio, 0)

    return merged_image


def merge_colored(img1, img2, ratio, threshold=0.5):
    """
    Merge an RGB image with a grayscale image after converting the grayscale image to pseudo-color.
    High values in the grayscale image are filled with orange, and low values are transparent.

    Parameters:
    img1 (numpy.ndarray): The first input image in RGB format.
    img2 (numpy.ndarray): The second input image in grayscale format.
    ratio (float): The ratio for blending the two images.
    threshold (float): The threshold to determine high and low values.

    Returns:
    numpy.ndarray: The merged image.
    """
    # Normalize img2 to range [0, 1]
    normalized_img2 = img2 / img2.max()

    # Create an empty image with the same size as img1, but with an alpha channel
    pseudo_color = np.zeros((img2.shape[0], img2.shape[1], 4), dtype=np.uint8)

    # Fill high values with varying intensity of orange (BGR color) and alpha = 255 (fully opaque)
    orange = np.array([0, 69, 255], dtype=np.uint8)
    for i in range(pseudo_color.shape[0]):
        for j in range(pseudo_color.shape[1]):
            if normalized_img2[i, j] > threshold:
                intensity = normalized_img2[i, j]
                pseudo_color[i, j, :3] = intensity * orange
                pseudo_color[i, j, 3] = 255
            else:
                pseudo_color[i, j, 3] = 0

    # Convert the grayscale image to 3-channel RGB image for merging
    img1_with_alpha = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)

    # Resize images to the same size if they are not already
    height, width, _ = img1.shape
    pseudo_color_resized = cv2.resize(pseudo_color, (width, height))

    # Merge images with the given ratio, considering alpha channel
    merged_image = cv2.addWeighted(img1_with_alpha, ratio, pseudo_color_resized, 1 - ratio, 0)

    # Convert back to 3-channel RGB image
    merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGRA2BGR)

    return merged_image


