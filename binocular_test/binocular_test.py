import cv2
import numpy as np

# 加载左右相机拍摄的图像
left_img = cv2.imread('left_image.jpg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right_image.jpg', cv2.IMREAD_GRAYSCALE)

# 假设相机内参和外参已经通过校准获得
K1 = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])  # 左相机内参矩阵
K2 = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])  # 右相机内参矩阵
D1 = np.zeros(5)  # 左相机畸变系数
D2 = np.zeros(5)  # 右相机畸变系数
R = np.eye(3)  # 假设两个相机的旋转矩阵为单位矩阵
T = np.array([[0.1], [0], [0]])  # 假设两个相机间的平移向量

# 立体校正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, left_img.shape[:2], R, T)
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, left_img.shape[:2], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, right_img.shape[:2], cv2.CV_32FC1)

left_rectified = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

# 计算视差图
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

# 将视差图转换为深度图（假设基线和焦距已知）
focal_length = 700  # 假设焦距为700像素
baseline = 0.1  # 假设基线为0.1米
depth_map = (focal_length * baseline) / (disparity + 1e-6)  # 防止除零

# 生成中间视角的图像
# 使用简单的线性插值方法生成中间视角图像
alpha = 0.5  # 中间视角的权重
middle_view = cv2.addWeighted(left_rectified, alpha, right_rectified, 1 - alpha, 0)

# 显示结果图像
cv2.imshow('Left Image', left_img)
cv2.imshow('Right Image', right_img)
cv2.imshow('Left Rectified', left_rectified)
cv2.imshow('Right Rectified', right_rectified)
cv2.imshow('Disparity', disparity / np.max(disparity))  # 归一化显示视差图
cv2.imshow('Middle View', middle_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
