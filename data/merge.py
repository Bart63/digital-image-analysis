import glob
import shutil
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def merge_testset():
    main_images = glob.glob(os.path.join(BASE_DIR, 'ImageNetV2-top-images/*/*'))
    main_images = [os.path.basename(x) for x in main_images]

    output_dir = os.path.join(BASE_DIR, 'ImageNetV2-test')
    os.makedirs(output_dir, exist_ok=True)

    rest_imgs1 = glob.glob(os.path.join(BASE_DIR, 'ImageNetV2-matched-frequency/*/*'))
    rest_imgs2 = glob.glob(os.path.join(BASE_DIR ,'ImageNetV2-threshold-0.7/*/*'))

    rest_imgs1 = list(filter(lambda fn: os.path.basename(fn) not in main_images, rest_imgs1))
    rest_imgs2 = list(filter(lambda fn: os.path.basename(fn) not in main_images, rest_imgs2))
    rest_paths = rest_imgs1 + rest_imgs2

    output_basenames = []

    for path in rest_paths:
        basename = os.path.basename(path)
        if basename in output_basenames:
            continue
        output_basenames.append(basename)
        idx_path = path.split('/')[-2]
        os.makedirs(os.path.join(output_dir, idx_path), exist_ok=True)
        output_imgs = os.path.join(output_dir, idx_path, basename)
        shutil.copy(path, output_imgs)
    
    shutil.rmtree(os.path.join(BASE_DIR, 'ImageNetV2-matched-frequency'))
    shutil.rmtree(os.path.join(BASE_DIR, 'ImageNetV2-threshold-0.7'))
