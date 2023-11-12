import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity as cs


def ssim_similarity(img1:np.ndarray, img2:np.ndarray):
    if img1.dtype != np.uint8 or img2.dtype != np.uint8:
        val_max = np.maximum(img1, img2)
        val_min = np.minimum(img1, img2)
        return ssim(img1, img2, data_range=val_max - val_min)
    return ssim(img1, img2)


def mse_similarity(img1:np.ndarray, img2:np.ndarray):
    return 1 / mse(img1, img2)


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
    'cosine': cosine_similarity,
    'bhattacharyya': bhattacharyya_distance
}
