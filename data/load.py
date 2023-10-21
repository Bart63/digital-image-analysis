from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import Dataset

from .ImageNetV2TestDataset import ImageNetV2TestDataset
from .consts import TYPES, IMAGENET_2012_LABELS

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(typename="top-images") -> (Dataset, dict):
    if typename not in TYPES:
        raise ValueError(f'Invalid typename: {typename}. Available types: {TYPES}')
    
    ds = ImageNetV2Dataset(typename, location=BASE_DIR)
    ds_test = ImageNetV2TestDataset(location=BASE_DIR)
    return ds, ds_test, IMAGENET_2012_LABELS
