import os
from os import listdir
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import logging

# Setting up the log file
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')


class DoodleDatasetSimple(Dataset):
    """
    Class that prepares the dataset for loading
    :param doodle_path: The path where the images are located
    :param transform: The transformation to be applied to each image
    :param translation: The dictionary to match each of the images to its label
    :return An image ready to be fed into the model and its corresponding class label
    """
    def __init__(self, doodle_path, transform, translation):
        try:
            self.path = doodle_path
            self.folder = [x for x in listdir(doodle_path)]
            self.transform = transform
            self.translation_dict = translation
        except Exception as e:
            logging.error(f"Error in __init__: {e}")
            raise
        
    def __len__(self):
        try:
            return len(self.folder)
        except Exception as e:
            logging.error(f"Error in __len__: {e}")
            raise
        
    def __getitem__(self, idx):
        try:
            img_loc = os.path.join(self.path, self.folder[idx])
            image = Image.open(img_loc).convert('RGB')
            single_img = self.transform(image)
        except Exception as e:
            logging.error(f"Error in __getitem__ during image loading and transformation: {e}")
            raise
        
        try:
            imageClass = self.translation_dict[self.folder[idx]]
            sample = {'image': torch.from_numpy(np.array(single_img)),
                    'class': imageClass}
        except Exception as e:
            logging.error(f"Error in __getitem__ during class creation: {e}")
            raise
        
        return sample

