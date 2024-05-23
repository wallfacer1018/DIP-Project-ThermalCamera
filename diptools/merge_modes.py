import cv2
import frequency
import regulator

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


def merge_colored(img1, img2, ratio):
    """
    Merge an RGB image with a grayscale image after converting the grayscale image to pseudo-color.

    Parameters:
    img1 (numpy.ndarray): The first input image in RGB format.
    img2 (numpy.ndarray): The second input image in grayscale format.
    ratio (float): The ratio for blending the two images.

    Returns:
    numpy.ndarray: The merged image.
    """
    # Convert the grayscale image to pseudo-color using a colormap
    pseudo_color = cv2.applyColorMap(cv2.convertScaleAbs(img2, alpha=255.0 / img2.max()), cv2.COLORMAP_JET)

    # Resize images to the same size if they are not already
    height, width, _ = img1.shape
    pseudo_color_resized = cv2.resize(pseudo_color, (width, height))

    # Merge images with the given ratio
    merged_image = cv2.addWeighted(img1, ratio, pseudo_color_resized, 1 - ratio, 0)

    return merged_image

