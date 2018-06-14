# coding = utf-8

from __future__ import print_function
import cv2
import numpy as np
import tempfile
from math import ceil
import caffe


VIDEO = "/home/wangbin/PycharmProjects/hello_charm/mid.mp4"
FLOW_FILE = "/home/wangbin/PycharmProjects/hello_charm/midpredict.flo"
caffemodel = "/home/wangbin/github/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5"
deployproto = "/home/wangbin/github/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template"
WRITER = "/home/wangbin/PycharmProjects/hello_charm/midpredict.avi"
PREDICT_PNG = "/home/wangbin/PycharmProjects/hello_charm/predict.png"


def load_model(width, height):
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width / divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height / divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH'])
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT'])

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    proto = open(deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
        tmp.write(line)
    tmp.flush()

    caffe.set_logging_disabled()
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, caffemodel, caffe.TEST)
    print('Network forward pass using %s.' % caffemodel)

    return net


def video2flow():
    cap = cv2.VideoCapture(VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    fps = 8.0
    writer = cv2.VideoWriter(WRITER, fourcc, fps, (width/2, height))

    _, frame1 = cap.read()
    _, frame2 = cap.read()

    net = load_model(width, height)
    while True:
        # get a frame
        frame1 = frame2
        _, frame2 = cap.read()
        if frame2 is None:
            break
        blob = predict_flow(frame1, frame2, net)
        flow = visualize_optical_flow(frame1, blob)
        hmerge = np.vstack((frame1, flow))
        hmerge = cv2.resize(hmerge, (width/2, height))
        cv2.imshow("frame-flow", hmerge)
        writer.write(hmerge)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def predict_flow(frame0, frame1, net):

    num_blobs = 2
    input_data = [frame0[np.newaxis, :, :, :].transpose(0, 3, 1, 2),
                  frame1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)]  # batch, bgr, h, w

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

    # There is some non-deterministic nan-bug in caffe
    i = 1
    while i <= 5:
        i += 1
        net.forward(**input_dict)
        contains_NaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                contains_NaN = True

        if not contains_NaN:
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)  # CHW -> HWC
    return blob


def visualize_optical_flow(frame1, blob):
    # optical flow visualization
    hsv = np.zeros_like(frame1)
    rad, ang = cv2.cartToPolar(blob[..., 0], blob[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(rad, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    gr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return gr


if __name__ == "__main__":
    print("start predict optical flow")
    video2flow()
