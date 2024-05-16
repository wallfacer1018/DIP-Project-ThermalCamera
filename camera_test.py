# -*- coding: UTF-8 -*-
# 引入Picamera2模块和时间模块
from picamera2 import Picamera2
from time import sleep

# 创建Picamera2对象
picam2 = Picamera2()

# 配置摄像头参数
camera_config = picam2.create_preview_configuration(main={"size": (1024, 768)})
picam2.configure(camera_config)

# 启动摄像头预览
picam2.start()

# 预热，让摄像头适应光线
sleep(2)

# 拍照并存储文件
print('正在拍照...')
picam2.capture_file('1.jpg')
print('完成!')
picam2.stop()