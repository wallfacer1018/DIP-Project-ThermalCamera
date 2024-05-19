import pygame
import pygame.camera
from pygame.locals import *

# 初始化pygame和摄像头模块
pygame.init()
pygame.camera.init()

# 获取可用摄像头列表
camlist = pygame.camera.list_cameras()
if len(camlist) < 2:
    print("Error: Less than two cameras found!")
    exit()

# 打开两个摄像头
cam1 = pygame.camera.Camera(camlist[0], (640, 480))
cam2 = pygame.camera.Camera(camlist[1], (640, 480))
cam1.start()
cam2.start()

# 创建用于显示摄像头图像的窗口
screen1 = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Camera 1 Preview")
screen2 = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Camera 2 Preview")

# 创建Surface用于捕捉图像
snapshot1 = pygame.Surface((640, 480), 0, screen1)
snapshot2 = pygame.Surface((640, 480), 0, screen2)

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # 获取摄像头图像
    snapshot1 = cam1.get_image(snapshot1)
    snapshot2 = cam2.get_image(snapshot2)

    # 显示图像
    screen1.blit(snapshot1, (0, 0))
    pygame.display.flip()
    screen2.blit(snapshot2, (0, 0))
    pygame.display.flip()

# 关闭摄像头
cam1.stop()
cam2.stop()
pygame.quit()
