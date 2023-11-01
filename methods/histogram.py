import cv2
import numpy as np


def hist_normalize(image:np.ndarray):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    normalized_hist = hist / hist.sum()
    return normalized_hist
