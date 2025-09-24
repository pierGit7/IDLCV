"""
Dataset class for Hotdog/Not Hotdog classification.
"""

import os
import glob
import torch
import PIL.Image as Image


class Hotdog_NotHotdog(torch.utils.data.Dataset):
    """Dataset class for loading hotdog/not hotdog images."""
    
    def __init__(self, train, transform, data_path):
        """
        Initialize the dataset.
        
        Args:
            train (bool): If True, load training data, else load test data
            transform (torchvision.transforms): Transformations to apply to images
            data_path (str): Path to the dataset directory
        """
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        print(data_path)
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is the transformed PIL image and label is the class index
        """
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y
