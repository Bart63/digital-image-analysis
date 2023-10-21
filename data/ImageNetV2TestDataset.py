import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImageNetV2TestDataset(Dataset):
    def __init__(self, transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/ImageNetV2-test/")
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform
        if not self.dataset_root.exists():
            raise Exception('Dataset imagenet-test not found on disk')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
