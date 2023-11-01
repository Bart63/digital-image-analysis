import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity as cs


def ssim_similarity(image1:np.ndarray, image2:np.ndarray):
    if image1.dtype != np.uint8 or image2.dtype != np.uint8:
        val_max = np.maximum(image1, image1)
        val_min = np.minimum(image1, image1)
        return ssim(image1, image2, data_range=val_max - val_min)
    return ssim(image1, image2)


def mse_similarity(image1:np.ndarray, image2:np.ndarray):
    return 1 / mse(image1, image2)


def cosine_similarity(v1:np.ndarray, v2:np.ndarray):
    return cs(v1, v2)[0][0]


METRICS = {
    'ssim': ssim_similarity,
    'mse': mse_similarity,
    'cosine': cosine_similarity
}
