# -*- coding: UTF-8 -*-
#引入时间模块和摄像头模块
from time import sleep
from picamera import PiCamera

#设置摄像头参数
camera = PiCamera()
camera.resolution = (1024,768)
camera.start_preview()
#预热
sleep(2)
#拍照，存入文件
print('taking photo...')
camera.capture('2.jpg')
print('done!')
camera.stop_preview()