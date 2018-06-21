# coding=utf-8
from __future__ import print_function
import cv2
import numpy as np

VIDEO = "/home/wangbin/PycharmProjects/hello_charm/midpredict_KITTI.avi"
WRITER = "/home/wangbin/PycharmProjects/hello_charm/video/edge_KITTI_gauss_50_25.avi"
PNG = "/home/wangbin/PycharmProjects/hello_charm/predict.png"


def video2flow():
    cap = cv2.VideoCapture(VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    fps = 8.0
    writer = cv2.VideoWriter(WRITER, fourcc, fps, (width, height))

    while True:
        # get a frame
        _, frame = cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur_edge = cv2.blur(frame, (3, 3))
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        edge = cv2.Canny(blur, 50, 25, 3)  # Canny算子处理之后的是灰度图, 单通道
        mask = np.where(edge > 0, 1, 0).astype(np.uint8)  # 获取掩码, 图像的数据类型为 unsigned char
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 单通道转3通道
        edge_bgr = np.multiply(mask, frame)

        cv2.imshow("original frame", frame)
        cv2.imshow("blur", blur)
        cv2.imshow("canny edge detect", edge_bgr)

        writer.write(edge_bgr)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    video2flow()
