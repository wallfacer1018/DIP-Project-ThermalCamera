import numpy as np
import matplotlib.pyplot as plt
from diptools import *
import cv2

# Function to find the proper left distance of the selected area of the visible image
# 在物体比较远，几乎没有偏移的情况下，如res_findleft中的visible0.jpg、thermal0.jpg，合适的left值约为130
# 在物体比较近，如res_findleft中的visible1.jpg、thermal1.jpg，合适的left值约为160
def find_left(visible, thermal, sigma_visible, sigma_thermal, scale):
    relation = np.zeros(250)
    thermal = thermal[0:240, 60:300]
    thermal = cv2.resize(thermal, (int(240 * scale), int(240 * scale)), interpolation=cv2.INTER_CUBIC)
    H_p = frequency.H_gauss_LPF(thermal.shape, int(sigma_thermal*scale), double_pad=True)
    H_p = 1 - H_p
    thermal_highpass = frequency.filter_freq_pad(thermal, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)
    for i in range(125, 225):
        visible_tmp = visible[0:768, i:i+768]
        visible_tmp = cv2.resize(visible_tmp, (int(240*scale), int(240*scale)), interpolation=cv2.INTER_CUBIC)
        H_p = frequency.H_gauss_LPF(visible_tmp.shape, sigma_visible*scale, double_pad=True)
        H_p = 1 - H_p
        visible_tmp_highpass = frequency.filter_freq_pad(visible_tmp, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)

        relation[i] = np.sum(visible_tmp_highpass * thermal_highpass)
    # 绘制数组值
    plt.plot(relation, marker='o', linestyle='-', color='b', label='Array Values')

    # 添加标题和标签
    plt.title('Array Values Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图像
    plt.show()
    print(np.argmax(relation))
    return np.argmax(relation)

visible_image = persistence.load_gray('res_findleft/visible0.jpg')
thermal_image = persistence.load_gray('res_findleft/thermal0.jpg')

length = 768
left = find_left(visible_image, thermal_image, 30, 30, 0.3)
top = 0
right = left + length
bottom = int(top + length)
visible_image = visible_image[top:bottom, left:right]

visible_image = cv2.resize(visible_image, (240, 240), interpolation=cv2.INTER_CUBIC)
thermal_normalized = regulator.GrayScalingRegulator(thermal_image)
thermal_normalized = thermal_normalized[0:240, 60:300]
combined_image = merge_modes.merge_grayscale(visible_image, thermal_normalized, 0.5)

persistence.show(combined_image)