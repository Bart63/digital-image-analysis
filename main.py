import numpy as np
from data import load_data
from filter import filter_images
from itertools import product
from scale_crop import scale_crop


def calc_similarity(indexes=[1,8,58], sim_fn=None):
    ds, ds_test, idx2label = load_data()
    
    # Filter images with label indexes
    ds = filter_images(ds, indexes)
    ds_test = filter_images(ds_test, indexes)

    print(len(ds), len(ds_test))

    # Iterate over all pairs
    ds_prod = product(ds, ds_test)
    for ds1, ds2 in ds_prod:
        img1, lab1 = ds1
        img2, lab2 = ds2
        
        img1, img2 = scale_crop(img1, img2)
        img1, img2 = np.array(img1), np.array(img2)

        similarity = sim_fn(img1, img2)
        print(lab1, lab2, similarity)


def main():
    calc_similarity()


if __name__ == '__main__':
    main()
