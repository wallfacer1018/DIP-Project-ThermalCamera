import pygame
import pygame.camera
from pygame.locals import *

# 初始化pygame和摄像头模块
pygame.init()
pygame.camera.init()

# 获取可用摄像头列表
camlist = pygame.camera.list_cameras()
if not camlist:
    print("No camera found!")
    exit()

# 打开第一个摄像头
cam = pygame.camera.Camera(camlist[0], (640, 480))
cam.start()

# 创建用于显示摄像头图像的窗口
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("USB Camera Preview")

# 创建一个Surface用于捕捉图像
snapshot = pygame.Surface((640, 480), 0, screen)

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # 获取摄像头图像
    snapshot = cam.get_image(snapshot)

    # 显示图像
    screen.blit(snapshot, (0, 0))
    pygame.display.flip()

# 关闭摄像头
cam.stop()
pygame.quit()
