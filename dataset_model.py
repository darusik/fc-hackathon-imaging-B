import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from medmnist.info import INFO

# Slight Adaption of the MedMnist Class in medmnist
class CustomChestMnist(Dataset):

    flag = "chestmnist"

    def __init__(self,
                 imgs=None, 
                 labels=None,
                 transform=None,
                 target_transform=None,
                ):
        ''' dataset
        :param imgs: img data
        :param labels: respective labels
        :param transform: data transformation
        :param target_transform: target transformation
        '''

        self.info = INFO[self.flag]

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __repr__(self):
        '''Adapted from torchvision.ss'''
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)
