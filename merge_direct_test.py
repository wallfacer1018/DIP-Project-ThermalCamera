import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from diptools import regulator

# 设置文件路径
res_folder = "res"
camera_12_path = os.path.join(res_folder, "camera_12.jpg")
them_path = os.path.join(res_folder, "them.jpg")

# 读取图像
camera_12 = Image.open(camera_12_path)
them = Image.open(them_path)

# 将them.jpg转化为灰度图像并左右翻转
them_gray = them.convert("L")
them_flipped = them_gray.transpose(Image.FLIP_LEFT_RIGHT)

# 将camera_12.jpg转化为灰度图像
camera_12_gray = camera_12.convert("L")

# 裁剪出中间一块
width, height = camera_12_gray.size
length = 700
left = 160
top = 160
right = left + length
bottom = top + length * 3 / 4
camera_12_cropped = camera_12_gray.crop((left, top, right, bottom))

# 将裁剪后的图像转化为NumPy数组
camera_12_cropped_np = np.array(camera_12_cropped)

# Bicubic插值到480x640
camera_12_resized = scipy.ndimage.zoom(camera_12_cropped_np, (480 / camera_12_cropped_np.shape[0], 640 / camera_12_cropped_np.shape[1]), order=3)
them_flipped_resized = them_flipped.resize((640, 480), Image.BICUBIC)

# 灰度图相加
factor=0.3
merged = factor * camera_12_resized + np.array(them_flipped_resized)*(1-factor)
merged = regulator.GrayCuttingRegulator(merged)

# 显示图像
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(them_flipped_resized, cmap='gray')
axes[0].set_title('Them (Flipped and Resized)')
axes[0].axis('off')

# 使用伪彩色映射显示 merged
axes[1].imshow(merged, cmap='viridis')
axes[1].set_title('Camera 12 (Cropped, Resized, and Merged)')
axes[1].axis('off')

plt.show()
