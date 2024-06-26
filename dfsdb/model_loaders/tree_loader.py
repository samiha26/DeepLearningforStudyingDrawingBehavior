import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
import logging

# Setting up log file
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')


class MultilabelClassifier(nn.Module):
    """
    Initialize the model architecture
    Exactly the same as the one used in the classifier
    """
    def __init__(self, n_features):
        super().__init__()
        try:
            self.resnet = models.resnet34(pretrained=True)
            self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

            self.imageClass = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(in_features=512, out_features=n_features)
            )
        except Exception as e:
            logging.error("Error initializing MultilabelClassifier: %s", e)
            raise
        
    def forward(self, x):
        try:
            x = self.model_wo_fc(x)
            x = torch.flatten(x, 1)

            return {
                'class': self.imageClass(x)
            }
        except Exception as e:
            logging.error("Error in forward pass function of MultilabelClassifier: %s", e)
            raise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def predict(modelPath, imagePath):
    """
    Method that loads the trained tree model from file
    Gives an output label for the image in the path that is specified
    :param modelPath: The path to the model to be used
    :param imagePath: The path to the image to be classified
    :return: The label of the image as predicted by the model
    """
    try:
        model = MultilabelClassifier(3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        checkpoint = torch.load(modelPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        model.eval()

        img_loc = imagePath
        raw_img = Image.open(img_loc).convert('RGB')
        single_img = data_transforms(raw_img)
        single_img = single_img.unsqueeze(0)
        tmp = torch.from_numpy(np.array(single_img))
        outputs = model(tmp.to(device))
        res = 0
        for i, out in enumerate(outputs):
            _, predicted = torch.max(outputs[out], 1)
            res = predicted.item()

        return res
    except Exception as e:
        logging.error("Error in prediction: %s", e)
        raise