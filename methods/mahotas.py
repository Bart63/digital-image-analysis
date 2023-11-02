import numpy as np
import mahotas.features as mf
from .metrics import cosine_similarity


def haralick_similarity(img1:np.ndarray, img2:np.ndarray, distance=1):
    v1 = mf.haralick(img1, return_mean=True, distance=distance).reshape(1, -1)
    v2 = mf.haralick(img2, return_mean=True, distance=distance).reshape(1, -1)
    return cosine_similarity(v1, v2)
