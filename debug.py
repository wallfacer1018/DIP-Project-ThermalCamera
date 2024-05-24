import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
from diptools import *
import cv2


image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
image = fisheye_correction.fisheye_correction(image)
# select certain area of the image

width, height = image.shape
length = 700
left = 0
top = 0
right = left + length
bottom = int(top + length * 3 / 4)
image = image[top:bottom, left:right]

image_resized = cv2.resize(image, (240, 320), interpolation=cv2.INTER_CUBIC)

persistence.show(image_resized)