# coding=utf-8
import cv2
import numpy as np

img1 = cv2.imread("/home/wangbin/PycharmProjects/hello_charm/predict.png")
img2 = cv2.imread("/home/wangbin/PycharmProjects/hello_charm/predict.png")

# resize to same scale
im1 = cv2.resize(img1, (400, 400))
im2 = cv2.resize(img2, (400, 400))
hmerge = np.hstack((im1, im2)) #水平拼接
# vmerge = np.vstack((im1, im2)) #垂直拼接

cv2.imshow("test1", hmerge)
# cv2.imshow("test2", vmerge)

cv2.waitKey(0)
cv2.destroyAllWindows()
