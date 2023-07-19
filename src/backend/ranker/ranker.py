import pickle
import numpy as np
# import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("../logs/ranker.log"),
                              logging.StreamHandler()])

logger = logging.getLogger('ranker')


class Ranker():
    def __init__(self,
                 model_path: str,
                 K: int = 5):

        with open(model_path, 'rb') as model:
            self.model = pickle.load(model)

        self.K = K
        logger.info(f"Ranker initialized from {model_path}")
        logger.info(f"{K=}")
        logger.info(f'RANKER={self.model}')

    def rank(self, query: np.ndarray) -> np.ndarray:
        logger.info(f"query.shape={query.shape}")
        query = query.reshape(1, -1)

        distances, indices = self.model.kneighbors(query, n_neighbors=self.K)
        logger.info(f"query={query} => indices={indices}")

        return distances.reshape(-1), indices.reshape(-1)
