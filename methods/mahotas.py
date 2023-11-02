import numpy as np
import mahotas.features as mf
from .metrics import cosine_similarity


def haralick_similarity(image1:np.ndarray, image2:np.ndarray, distance=1):
    v1 = mf.haralick(image1, return_mean=True, distance=distance).reshape(1, -1)
    v2 = mf.haralick(image2, return_mean=True, distance=distance).reshape(1, -1)
    return cosine_similarity(v1, v2)
