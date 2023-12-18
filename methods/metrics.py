import math
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity as cs


def ssim_similarity(img1:np.ndarray, img2:np.ndarray):
    channel_axis = 2 if img2.ndim == 3 else None
    if img1.dtype != np.uint8 or img2.dtype != np.uint8:
        val_max = np.maximum(img1, img2)
        val_min = np.minimum(img1, img2)
        return ssim(img1, img2, data_range=val_max - val_min, channel_axis=channel_axis)
    return ssim(img1, img2, channel_axis=channel_axis)


def mse_similarity(img1:np.ndarray, img2:np.ndarray):
    return 1 / mse(img1, img2)

def psnr(img1:np.ndarray, img2:np.ndarray):
    similarity = mse(img1, img2)
    peak=np.max(img1)**2
    ratio=10*math.log10(peak*similarity)
    return ratio

def cosine_similarity(v1:np.ndarray, v2:np.ndarray):
    return cs(v1, v2)[0][0]

def bhattacharyya_distance(hist1, hist2):
    hist1 = np.ravel(hist1)
    hist2 = np.ravel(hist2)

    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    b_coefficient = np.sum(np.sqrt(hist1 * hist2))

    b_distance = -np.log(b_coefficient)

    return b_distance

METRICS = {
    'ssim': ssim_similarity,
    'mse': mse_similarity,
    'psnr': psnr,
    'cosine': cosine_similarity,
    'bhattacharyya': bhattacharyya_distance,
}
