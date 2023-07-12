from torch import no_grad, rand
# import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import mlflow
import numpy as np
import os
import logging

REGISTRY_URI = 'file:///home/zeinovich/projects/image-ranking/mlflow'
STAGE = 'Staging'
PATH = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f"{PATH}/../../../logs/extractor.log"),
                                logging.StreamHandler()])


logger = logging.getLogger(__name__)
logger.info(f'{REGISTRY_URI=}')
logger.info(f'{STAGE=}')

mlflow.set_registry_uri(REGISTRY_URI)
MODEL_PATH = f'models:/FeatureExtractor/{STAGE}'

logger.info(f'{MODEL_PATH=}')

class FeatureExtractor():
    def __init__(self,
                 model_path: str,
                 device: str='cuda'):
        
        self.device = device
        self.model = mlflow.pytorch.load_model(model_path,
                                               dst_path=f'{PATH}/../../../tmp/extractor').to(self.device)
        self.transform = Compose([ToTensor()])
        self.model.eval()

    @no_grad()
    def extract(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(image).cpu().numpy().flatten()

    def _get_output_shape(self, image_dim=(1, 3, 100, 100)):
        return self.model(rand(*(image_dim))).data.shape

