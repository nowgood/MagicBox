def visualize_optical_flow(frame1, blob):
    # optical flow visualization
    hsv = np.zeros_like(frame1)
    rad, ang = cv2.cartToPolar(blob[..., 0], blob[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(rad, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow', rgb)
