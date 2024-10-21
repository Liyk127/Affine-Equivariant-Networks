# Adapted from https://github.com/rghosh92/SS-CNN


import os, pickle
import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset_name, inputs, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        'Denotes the total number of samples'
        return self.inputs.shape[0]

    def cutout(self, img, x, y, size):
        size = int(size/2)
        lx = np.maximum(0, x-size)
        rx = np.minimum(img.shape[0], x+size)
        ly = np.maximum(0, y - size)
        ry = np.minimum(img.shape[1], y + size)
        img[lx:rx,ly:ry,:] = 0
        return img

    def __getitem__(self, index):
        'Generates one sample of data'
        img = self.inputs[index]
        if self.transform is not None:
            img = self.transform(img)
        y = int(self.labels[index])
        return img, y


def load_dataset(dataset_name, val_splits, training_size):

    os.chdir(dataset_name)
    a = os.listdir()
    listdict = []

    for split in range(val_splits):

        listdict.append(pickle.load(open(a[split], 'rb')))

        listdict[-1]['train_data'] = np.float32(listdict[-1]['train_data'][0:training_size, :, :])
        listdict[-1]['train_label'] = listdict[-1]['train_label'][0:training_size]

        listdict[-1]['test_data'] = np.float32(listdict[-1]['test_data'])
        listdict[-1]['test_label'] = np.float32(listdict[-1]['test_label'])

    os.chdir('..')
    os.chdir('..')

    return listdict