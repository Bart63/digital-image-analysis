import cv2
import numpy as np


def hist_normalize(img:np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    normalized_hist = hist / hist.sum()
    return normalized_hist
