import cv2
import numpy as np

# 加载图片
image = cv2.imread('res/fisheye_image.jpg')

# 鱼眼相机的参数（需根据你的相机参数进行调整）
DIM = (image.shape[1], image.shape[0])  # 图像的尺寸
K = np.array([[700, 0, DIM[0] / 2], [0, 700, DIM[1] / 2], [0, 0, 1]])  # 内参矩阵
D = np.array([-0.2, 0.1, 0, 0])  # 畸变系数

# 去畸变
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# 显示结果
cv2.imshow('undistorted_image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
