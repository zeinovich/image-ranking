import mlflow
import numpy as np
import os
import logging

REGISTRY_URI = 'file:///home/zeinovich/projects/image-ranking/mlflow'
STAGE = 'Staging'
MODEL_PATH = f'models:/Ranker/{STAGE}'
PATH = os.path.dirname(os.path.abspath(__file__))

mlflow.set_registry_uri(REGISTRY_URI)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f"{PATH}/../../../logs/ranker.log"),
                                logging.StreamHandler()])

logger = logging.getLogger(__name__)


logger.info(f'{REGISTRY_URI=}')
logger.info(f'{STAGE=}')
logger.info(f'{MODEL_PATH=}')


class Ranker():
    def __init__(self,
                 model_path: str,
                 K: int=5):
        
        self.model = mlflow.sklearn.load_model(model_path,
                                               dst_path=f'{PATH}/../../../tmp/ranker')
        self.K = K

    def rank(self, query: np.ndarray) -> np.ndarray:
        _, indices = self.model.kneighbors(query, n_neighbors=self.K)
        return indices.reshape(-1)