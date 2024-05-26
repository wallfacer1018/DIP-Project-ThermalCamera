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

# Setup the plot
fig, ax = plt.subplots()
if pseudo_color:
    therm1 = ax.imshow(np.zeros((240, 240)), interpolation='none', cmap='inferno')
else:
    therm1 = ax.imshow(np.zeros((240, 240)), interpolation='none', cmap='gray')
cbar = fig.colorbar(therm1)
cbar.set_label('Intensity')


def capture_visible_image():
    # Capture image and directly return it as an array
    picam2.capture_file('visible.jpg')
    image = cv2.imread('visible.jpg', cv2.IMREAD_GRAYSCALE)
    # image = fisheye_correction.fisheye_correction(image)
    return image


def capture_thermal_image():
    try:
        mlx.getFrame(frame)
    except ValueError:
        return np.zeros((240, 320))
    thermal_data = np.reshape(frame, (24, 32))
    thermal_image_resized = cv2.resize(thermal_data, (320, 240), interpolation=cv2.INTER_CUBIC)
    return thermal_image_resized

# Function to find the proper left distance of the selected area of the visible image
def find_left(visible, thermal, sigma_visible, sigma_thermal, scale):
    relation = np.zeros(250)
    thermal = cv2.resize(thermal, (int(240 * scale), int(240 * scale)), interpolation=cv2.INTER_CUBIC)
    thermal_highpass = gauss_highpass(thermal, sigma_thermal)
    for i in range(125, 225):
        visible_tmp = visible[0:768, i:i+768]
        visible_tmp = cv2.resize(visible_tmp, (int(240*scale), int(240*scale)), interpolation=cv2.INTER_CUBIC)
        visible_tmp_highpass = gauss_highpass(visible_tmp, sigma_visible)
        relation[i] = np.sum(visible_tmp_highpass * thermal_highpass)
    print(np.argmax(relation))
    return np.argmax(relation)


# Function to update the plot
def update_fig(*args):
    visible_image = capture_visible_image()
    thermal_image = capture_thermal_image()
    thermal_image = np.fliplr(thermal_image)
    # persistence.save_gray('res_findleft/visible0.jpg', visible_image)
    # persistence.save_gray('res_findleft/thermal0.jpg', thermal_image)

    thermal_image = thermal_image[0:240, 60:300]

    length = 768
    left = find_left(visible_image, thermal_image, 30, 30, 0.3)
    top = 0
    right = left + length
    bottom = int(top + length)
    visible_image = visible_image[top:bottom, left:right]
    visible_image = cv2.resize(visible_image, (240,240), interpolation=cv2.INTER_CUBIC)
    if merge_highpass:
        visible_image = gauss_highpass(visible_image, 10)

    thermal_normalized = regulator.GrayScalingRegulator(thermal_image)
    combined_image = merge_modes.merge_grayscale(visible_image, thermal_normalized, 0.5)
    if flip:
        combined_image = np.fliplr(combined_image)
    therm1.set_array(combined_image)
    therm1.set_clim(vmin=np.min(combined_image), vmax=np.max(combined_image))
    return therm1


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
