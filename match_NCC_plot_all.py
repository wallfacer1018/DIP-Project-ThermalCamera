import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from picamera2 import Picamera2
from time import sleep
from diptools import *
import cv2

pseudo_color = True
flip = False
merge_highpass = True

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

# Setup the plot with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax_visible, ax_thermal, ax_relation, ax_combined = axes.flatten()

if pseudo_color:
    therm1 = ax_thermal.imshow(np.zeros((240, 240)), interpolation='none', cmap='inferno')
else:
    therm1 = ax_thermal.imshow(np.zeros((240, 240)), interpolation='none', cmap='gray')
cbar = fig.colorbar(therm1, ax=ax_thermal)
cbar.set_label('Intensity')

# Initialize other plots
img_visible = ax_visible.imshow(np.zeros((768, 1024)), cmap='gray')
img_relation = ax_relation.imshow(np.zeros(250), cmap='viridis')
img_combined = ax_combined.imshow(np.zeros((240, 240)), cmap='gray')

ax_visible.set_title('Visible Image')
ax_thermal.set_title('Thermal Image')
ax_relation.set_title('Relation Array')
ax_combined.set_title('Combined Image')


def capture_visible_image():
    picam2.capture_file('visible.jpg')
    image = cv2.imread('visible.jpg', cv2.IMREAD_GRAYSCALE)
    return image


def capture_thermal_image():
    try:
        mlx.getFrame(frame)
    except ValueError:
        return np.zeros((240, 320))
    thermal_data = np.reshape(frame, (24, 32))
    thermal_image_resized = cv2.resize(thermal_data, (320, 240), interpolation=cv2.INTER_CUBIC)
    return thermal_image_resized


def find_left(visible, thermal, scale):
    relation = np.ones(250) * -1
    thermal_resized = cv2.resize(thermal, (int(240 * scale), int(240 * scale)), interpolation=cv2.INTER_CUBIC)
    for i in range(100, 170):
        visible_tmp = visible[0:768, i:i+768]
        visible_tmp = cv2.resize(visible_tmp, (int(240 * scale), int(240 * scale)), interpolation=cv2.INTER_CUBIC)
        relation[i] = calculate_ncc(visible_tmp, thermal_resized)
    print(np.argmax(relation))
    return np.argmax(relation), relation


def calculate_ncc(reference_image, query_image):
    def calculate_mean(image):
        return np.mean(image)

    Ih, Iw = reference_image.shape
    Qh, Qw = query_image.shape

    mu_query = calculate_mean(query_image)
    reference_region = reference_image
    query_region = query_image
    mu_reference = calculate_mean(reference_region)

    numerator = np.sum((reference_region - mu_reference) * (query_region - mu_query))
    denominator = np.sqrt(
        np.sum((reference_region - mu_reference) ** 2) *
        np.sum((query_region - mu_query) ** 2)
    )

    if denominator != 0:
        ncc_value = numerator / denominator
    else:
        ncc_value = 0

    return ncc_value


def update_fig(*args):
    visible_image = capture_visible_image()
    thermal_image = capture_thermal_image()
    thermal_image = np.fliplr(thermal_image)
    thermal_image = thermal_image[0:240, 60:300]

    length = 768
    left, relation = find_left(visible_image, thermal_image, 0.3)
    top = 0
    right = left + length
    bottom = int(top + length)
    visible_image = visible_image[top:bottom, left:right]
    visible_image = cv2.resize(visible_image, (240, 240), interpolation=cv2.INTER_CUBIC)
    if merge_highpass:
        visible_image = gauss_highpass(visible_image, 10)

    thermal_normalized = regulator.GrayScalingRegulator(thermal_image)
    combined_image = merge_modes.merge_grayscale(visible_image, thermal_normalized, 0.5)
    if flip:
        combined_image = np.fliplr(combined_image)

    # Update plots
    img_visible.set_array(visible_image)
    img_relation.set_array(relation)
    img_combined.set_array(combined_image)
    therm1.set_array(thermal_image)

    therm1.set_clim(vmin=np.min(thermal_image), vmax=np.max(thermal_image))
    return img_visible, therm1, img_relation, img_combined


def gauss_highpass(img, sigma):
    H_p = frequency.H_gauss_LPF(img.shape, sigma, double_pad=True)
    H_p = 1 - H_p
    img_highpass = frequency.filter_freq_pad(img, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)
    return img_highpass


# Create an animation
ani = animation.FuncAnimation(fig, update_fig, interval=500)

# Show the plot
plt.show()

picam2.stop()
