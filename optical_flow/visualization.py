# coding=utf-8

def visualize_optical_flow(frame1, blob):
    # optical flow visualization
    # np.zeros_like(): Return an array of zeros with the same shape and type as a given array.
    hsv = np.zeros_like(frame1)
    # cv2.cartToPolar(x, y): brief Calculates the magnitude and angle of 2D vectors.
    rad, ang = cv2.cartToPolar(blob[..., 0], blob[..., 1])
    # degree to rad: degree*180/np.pi
    hsv[..., 0] = ang * 180 / np.pi
    # cv2.normalize: brief Normalizes the norm or value range of an array
    hsv[..., 1] = cv2.normalize(rad, None, 0, 255, cv2.NORM_MINMAX)
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
    hsv[..., 0] = ang * 180 / np.pi
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
