# coding=utf-8

def visualize_optical_flow(frame1, blob):
    # optical flow visualization 光流可视化
    # 由于frame的数据类型为np.uint8 即 usigned char, 最大存储值为255, 如果赋值为256, 结果为 0,
    # 也就是说及时赋值很大, 也会被截断
    # 对于 饱和度s 和亮度v 而言, 最大值是255, s = 255 色相最饱和, v = 255, 图片最亮
    # 而对与颜色而言, opencv3中, (0, 180) 就会把所有颜色域过一遍, 所以这里下面就算角度时会除以 2

    # np.zeros_like(): Return an array of zeros with the same shape and type as a given array.
    hsv = np.zeros_like(frame1)

    # cv2.cartToPolar(x, y): brief Calculates the magnitude and angle of 2D vectors.
    mag, ang = cv2.cartToPolar(blob[..., 0], blob[..., 1])

    # degree to rad: degree*180/np.pi
    hsv[..., 0] = (ang * 180 / np.pi) / 2
    
    # brief Normalizes the norm or value range of an array
    # norm_type = cv2.NORM_MINMAX, 即将值标准化到(0, 255)
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # 亮度为255
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def visualize_optical_flow(frame1, blob):
    # optical flow visualization
    # np.zeros_like(): Return an array of zeros with the same shape and type as a given array.
    hsv = np.zeros_like(frame1)
    # cv2.cartToPolar(x, y): brief Calculates the magnitude and angle of 2D vectors.
    rad, ang = cv2.cartToPolar(blob[..., 0], blob[..., 1])
    # degree to rad: degree*180/np.pi
    hsv[..., 0] = ang * 180 / np.pi / 2
    #
    hsv[..., 2] = cv2.normalize(rad, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 1] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
    
# 通过下面的代码理解上面的每一个函数  
import cv2
import numpy as np
x = np.arange(4, dtype=np.float64)
x = x.reshape((2, 2))
mag, ang = cv2.cartToPolar(x, x)
print("magnitude")
print(mag)
print("angle")
print(ang*180/np.pi)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
print("normlize mag")
print(mag)

'''
magnitude
[[0.         1.41421356]
 [2.82842712 4.24264069]]
angle
[[ 0.         44.99045634]
 [44.99045634 44.99045634]]
normlize mag
[[  0.  85.]
 [170. 255.]]
'''
