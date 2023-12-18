import json
import numpy as np
from tqdm import tqdm

from data import load_data
from scale_crop import scale_crop
import preprocess as prep

import methods as meth


SIM_FN = {
    'mse': meth.mse_similarity,
    'ssim': meth.ssim_similarity,
    'psnr': meth.ssim_similarity,
    'fourier': meth.fourier_similarity,
    'statistical': meth.stat_similarity,
    'haralick': meth.haralick_similarity,
    'hog': meth.hog_similarity,
    'magnitude-direction': meth.gradient_magnitude_and_direction,
    'lbp': meth.lbp_distance,
    'gabor': meth.gabor_similarity,
    'law': meth.law_similarity
}

def run_single_test(preprocess=[], sim_fn=SIM_FN['lbp']):
    ds, ds_test, idx2label = load_data()
    
    # Iterate over all pairs
    results = {}
    for ds_query in tqdm(ds_test):
        img_q, lab_q = ds_query
        query_fname = img_q.filename
        results[query_fname] = []
        for ds_key in ds:    
            img_k, lab_k = ds_key
            key_fname = img_k.filename
            
            img_q_sc, img_k_sc = scale_crop(img_q, img_k)
            img_q_np, img_k_np = np.array(img_q_sc), np.array(img_k_sc)

            for preprocess_fn in preprocess:
                img_q_np = preprocess_fn(img_q_np)
                img_k_np = preprocess_fn(img_k_np)
            similarity = sim_fn(img_q_np, img_k_np)
            results[query_fname].append(
                [key_fname, lab_q, lab_k, similarity]
            )
    return results



def run_multiple(preprocess, sim_fns, test_names):
    for test_name, sim_fn in zip(test_names, sim_fns):
        res = run_single_test(preprocess=preprocess, sim_fn=sim_fn)
        json.dump(res, open(f'results/{test_name}.json', 'w'))


def main():
    run_multiple(
        preprocess=[prep.rgb2gray],
        sim_fns=[
            SIM_FN['law'],
        ],
        test_names = ['law_max'],
    )


if __name__ == '__main__':
    main()
