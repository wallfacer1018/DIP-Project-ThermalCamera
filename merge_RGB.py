import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from picamera2 import Picamera2
from time import sleep
import cv2

# Initialize I2C bus and MLX90640 sensor
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)

print("MLX90640 addr detected on I2C")
print([hex(i) for i in mlx.serial_number])

mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (1024, 768)})
picam2.configure(camera_config)
picam2.start()
sleep(2)

# Create a frame buffer for MLX90640
frame = [0] * 768

# Setup the plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.set_title('Visible Image')
ax2.set_title('Thermal Image')
ax3.set_title('Merged Image')

therm1 = ax1.imshow(np.zeros((240, 320, 3), dtype=np.uint8))  # Ensure RGB format for initial image
therm2 = ax2.imshow(np.zeros((240, 320), dtype=np.uint8), cmap='jet', vmin=0, vmax=255)  # Thermal image
therm3 = ax3.imshow(np.zeros((240, 320, 3), dtype=np.uint8))  # Merged image

cbar = fig.colorbar(therm2, ax=ax2)
cbar.set_label('Intensity')

def capture_visible_image():
    # Capture image and directly return it as an array
    picam2.capture_file('visible.jpg')
    image = cv2.imread('visible.jpg')
    image_resized = cv2.resize(image, (240, 320))
    return image_resized

def capture_thermal_image():
    try:
        mlx.getFrame(frame)
    except ValueError:
        return np.zeros((24, 32))
    thermal_data = np.reshape(frame, (24, 32))
    return thermal_data

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
    pseudo_color_resized = cv2.resize(pseudo_color, (width, height), interpolation=cv2.INTER_NEAREST)

    # Merge images with the given ratio, considering alpha channel
    merged_image = cv2.addWeighted(img1_with_alpha, ratio, pseudo_color_resized, 1 - ratio, 0)

    # Convert back to 3-channel RGB image
    merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGRA2BGR)

    return merged_image

# Function to update the plot
def update_fig(*args):
    visible_image = capture_visible_image()
    thermal_image = capture_thermal_image()
    thermal_image_resized = cv2.resize(thermal_image, (240, 320), interpolation=cv2.INTER_CUBIC)
    combined_image = merge_colored(visible_image, thermal_image_resized, 0.2, 0.5)

    therm1.set_array(visible_image)
    therm2.set_array(thermal_image_resized)
    therm3.set_array(combined_image)

    therm2.set_clim(vmin=np.min(thermal_image_resized), vmax=np.max(thermal_image_resized))
    return therm1, therm2, therm3

# Create an animation
ani = animation.FuncAnimation(fig, update_fig, interval=500)

# Show the plot
plt.show()

picam2.stop()
