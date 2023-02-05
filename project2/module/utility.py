import numpy as np
import cv2


def getBbox(img1, M):
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    w_min, w_max, h_min, h_max = np.min(dst[:, 0, 0]), np.max(dst[:, 0, 0]), np.min(dst[:, 0, 1]), np.max(dst[:, 0, 1])
    return round(w_min), round(h_min), round(w_max), round(h_max)
