import numpy as np
import matplotlib.pyplot as plt
from .metrics import METRICS
import math

def psnr(img1:np.ndarray, img2:np.ndarray, metric='mse'):
    similarity = METRICS[metric](img1, img2)
    peak=(max(img1.reshape([img1.shape[0]*img1.shape[1]])))**2
    ratio=10*math.log10(peak*similarity)
    return ratio
