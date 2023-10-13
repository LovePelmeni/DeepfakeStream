from torch.utils import data
import numpy


class DeepFakeClassificationDataset(data.Dataset):

    def __init__(self, labels, images, transforms=None, weights=None):
        self.labels = labels
        self.images = images
        self.transforms = transforms
        self.weights = weights or numpy.ones(shape=len(numpy.unique(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx > len(self.images):
            raise IndexError("index is beyond dataset scope")
        label = self.labels[idx]
        img = self.images[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return label, img
