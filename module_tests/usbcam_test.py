import cv2

# 打开第一个摄像头，0表示第一个摄像头
cap1 = cv2.VideoCapture(0)
# 打开第二个摄像头，1表示第二个摄像头
cap2 = cv2.VideoCapture(1)

# 检查第一个摄像头是否成功打开
if not cap1.isOpened():
    print("Error: Could not open video device 0")
    exit()

# 检查第二个摄像头是否成功打开
if not cap2.isOpened():
    print("Error: Could not open video device 1")
    exit()

# 设置第一个摄像头的分辨率
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 设置第二个摄像头的分辨率
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 循环捕捉视频帧并显示
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        print("Error: Could not read frame from video device 0")
        break

    if not ret2:
        print("Error: Could not read frame from video device 1")
        break

    # 显示第一个摄像头的帧
    cv2.imshow('USB Camera 1', frame1)

    # 显示第二个摄像头的帧
    cv2.imshow('USB Camera 2', frame2)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap1.release()
cap2.release()
cv2.destroyAllWindows()
