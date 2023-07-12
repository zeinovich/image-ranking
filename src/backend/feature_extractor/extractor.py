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
                 device: str='cuda'):
        
        self.device = device
        self.model = torch.load(model_path, map_location=self.device)
        self.transform = Compose([ToTensor()])
        self.model.eval()

    @torch.no_grad()
    def extract(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(image).cpu().numpy().flatten().reshape(-1)

    def _get_output_shape(self, image_dim=(1, 3, 100, 100)):
        out = self.model(torch.rand(*(image_dim))).data
        out = out.cpu().numpy().flatten().reshape(-1)
        return out.shape
    
    @property
    def output_shape(self):
        return self._get_output_shape()
    
    def __call__(self, image):
        return self.extract(image)
    
    def __repr__(self):
        return f'FeatureExtractor(model={self.model}, device={self.device})'
    
    def __str__(self):
        return self.__repr__()

