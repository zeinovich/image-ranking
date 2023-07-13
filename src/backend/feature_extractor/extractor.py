import torch
# import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("../logs/extractor.log"),
                                logging.StreamHandler()])


logger = logging.getLogger(__name__)
class FeatureExtractor():
    def __init__(self,
                 model_path: str,
                 scaler=None, 
                 device: str='cpu'):
        
        '''
        model_path: path to the model
        scaler: scaler to be used to scale the output of the model
        device: device to be used for the model
        '''

        self.device = device
        self.model = torch.load(model_path, map_location=self.device)
        self.transform = Compose([ToTensor()])
        self.scaler = scaler
        self.model.eval()

        logger.info(f'Loaded model from {model_path}')
        logger.info(f'Using device {self.device}')
        logger.info(f'Using scaler {self.scaler}')
        logger.info(f'Using transform {self.transform}')

    @torch.no_grad()
    def extract(self, image):

        '''
        Extract features from the image
        image: image to be extracted
        
        returns: features of the image
        '''

        image = self.transform(image).unsqueeze(0).to(self.device)
        logger.info(f'Extracting features from image of shape {image.shape}')

        out = self.model(image).cpu().numpy().flatten().reshape(-1)
        logger.info(f'Extracted features of shape {out.shape}')

        if self.scaler is not None:
            out = self.scaler.transform(out)
            logger.info(f'Scaled features of shape {out.shape}')
        
        return out

    def _get_output_shape(self, image_dim=(1, 3, 100, 100)):

        '''
        Get the output shape of the model
        '''

        out = self.model(torch.rand(*(image_dim))).data
        out = out.cpu().numpy().flatten().reshape(-1)
        return out.shape
    
    @property
    def output_shape(self):
        '''
        Get the output shape of the model
        '''

        return self._get_output_shape()
    
    def __call__(self, image):
        return self.extract(image)
    
    def __repr__(self):
        return f'FeatureExtractor(model={self.model}, device={self.device})'
    
    def __str__(self):
        return self.__repr__()

