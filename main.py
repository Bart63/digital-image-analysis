import numpy as np
from itertools import product

from data import load_data
from filter import filter_images
from scale_crop import scale_crop
import preprocess as prep

import methods as meth


SIM_FN = {
    'mse': meth.mse_similarity,
    'ssim': meth.ssim_similarity,
    'fourier': meth.fourier_similarity,
    'statistical': meth.stat_similarity,
    'haralick': meth.haralick_similarity
}


def calc_similarity(indexes=[], preprocess=[], sim_fn=SIM_FN['haralick']):
    ds, ds_test, idx2label = load_data()
    
    # Filter images with label indexes
    ds = filter_images(ds, indexes)
    ds_test = filter_images(ds_test, indexes)

    # Iterate over all pairs
    ds_prod = product(ds, ds_test)
    for ds1, ds2 in ds_prod:
        img1, lab1 = ds1
        img2, lab2 = ds2
        
        img1, img2 = scale_crop(img1, img2)
        img1, img2 = np.array(img1), np.array(img2)

        for preprocess_fn in preprocess:
            img1 = preprocess_fn(img1)
            img2 = preprocess_fn(img2)

        similarity = sim_fn(img1, img2)
        print(lab1, lab2, similarity)


def main():
    calc_similarity(indexes=[1,8,58], preprocess=[prep.rgb2gray])


if __name__ == '__main__':
    main()
