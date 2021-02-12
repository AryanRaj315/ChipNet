from torch.utils.data import Dataset
import glob
import numpy as np
import pickle
import cv2
import os
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive

class ImageNet(Dataset):
    def __init__(self, root, train, transform, download=False):

        self.root = root
        self.root = os.path.join(self.root, "imagenet/ILSVRC/Data/CLS-LOC")
        self.train = train
        self.transform = transform
        with open('label_dict.pickle', 'rb') as f:
            self.label_dict = pickle.load(f)
        if train:
            self.paths = glob.glob(f"{self.root}/train/*/*")
        else:
            self.paths = glob.glob(f"{self.root}/val/*/*")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = pil_loader(self.paths[idx])
#         print(self.paths[idx])
        label = int(self.label_dict[self.paths[idx].split('/')[6]])
#         print(image.shape)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
        