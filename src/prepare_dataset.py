import os
from os import path
import torch
from torch.utils.data import Dataset
from skimage import io
from tqdm.auto import tqdm

from src.config import DATASET_PATH, TRANSFORM


class ImageDatasetLoaded(Dataset):
    def __init__(self, root, transform):
        """
        In case of small datasets or small image sizes,
        it is reasonable to speed up learning process by
        allocating all images into memory for faster retrieval

        :param root: root directory
        :param transform: trnsforms to apply for each image
        """
        self.images = torch.concat(
            [resolve_tensor(fname, transform) for fname in tqdm(os.listdir(root)) if
             path.isfile(path.join(root, fname))], dim=0)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)


class ImageDatasetNotLoaded(Dataset):
    def __init__(self, root, transform):
        """
        Usual images loading dataset with transforms

        :param root: root directory
        :param transform: trnsforms to apply for each image
        """
        self.root = root
        self.transform = transform
        self.fnames = list(filter(lambda x: path.isfile(path.join(root, x)), os.listdir(root)))

    def __getitem__(self, item):
        img = self.transform(io.imread(path.join(DATASET_PATH, self.fnames[item])))
        return img

    def __len__(self):
        return len(self.fnames)


def resolve_tensor(fname, transform):
    """
    Helper function for applying transforms when we load dataset to memory

    :param fname: filename
    :param transform: applied transforms
    :return: transformed tensor in the shape of [1, C, H, W]
    """
    return transform(io.imread(path.join(DATASET_PATH, fname))).unsqueeze(0)


def get_dataset(load_to_memory=False):
    if load_to_memory:
        return ImageDatasetLoaded(DATASET_PATH, TRANSFORM)

    return ImageDatasetNotLoaded(DATASET_PATH, TRANSFORM)
