from torch.utils import data
import typing
import cv2
class DeepFakeClassificationDataset(data.Dataset):
    """
    Class, responsible for managing 
    data, while training the network
    
    Parameters:
    -----------
    classes - true labels for a given set of imgs
    img_paths - list of img urls
    transformations - (Optional). augmentations for the images to apply
    """
    def __init__(self, classes: typing.List, img_paths: typing.List, transformations=None):
        self.classes = classes
        self.img_paths = img_paths
        self.transformations = transformations 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if idx > len(self.images):
            raise IndexError("index is beyond dataset scope")

        label = self.labels[idx]
        img = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)

        if self.transformations is not None:
            img = self.transformations(img)['image']

        return label, img
