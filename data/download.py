from imagenetv2_pytorch import ImageNetV2Dataset
from .consts import TYPES
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_dataset(typename="top-images"):
    if typename not in TYPES:
        raise ValueError(f'Invalid typename: {typename}. Available types: {TYPES}')
    ImageNetV2Dataset(typename, location=BASE_DIR) # supports matched-frequency, threshold-0.7, top-images variants
    archive = os.path.join(BASE_DIR, f'ImageNetV2-{typename}.tar.gz')
    if os.path.exists(archive):
        os.remove(archive)
    print(f'Dataset ImageNetV2-{typename} downloaded!')


def download_all():
    download_dataset('top-images')
    download_dataset('matched-frequency')
    download_dataset('threshold-0.7')
