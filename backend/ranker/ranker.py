import pickle
import numpy as np

# import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("./logs/ranker.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("ranker")


class Ranker:
    """
    Ranker class

    Parameters
    ----------
    model_path : str
        Path to the model
    K : int, optional
        Number of nearest neighbors to return, by default 5

    Attributes
    ----------
    model : sklearn.neighbors model
        Query model. Should have a `kneighbors` method
    K : int
        Number of nearest neighbors to return

    Methods
    -------
    rank(query: np.ndarray) -> np.ndarray
        Rank the query vector

    Examples
    --------
    >>> from ranker import Ranker
    >>> ranker = Ranker(model_path="model.pkl")
    >>> query = np.random.rand(1, 512)
    >>> distances, indices = ranker.rank(query)
    >>> distances.shape
    (5,)
    >>> indices.shape
    (5,)
    """

    def __init__(self, model_path: str, K: int = 5):

        with open(model_path, "rb") as model:
            self._model = pickle.load(model)

        self._K = K
        logger.info(f"Ranker initialized from {model_path}")
        logger.info(f"{self.K=}")
        logger.info(f"RANKER={self.model}")

    def rank(self, query: np.ndarray) -> np.ndarray:
        logger.info(f"query.shape={query.shape}")
        query = query.reshape(1, -1)

        distances, indices = self._model.kneighbors(query, n_neighbors=self.K)
        logger.info(f"query={query} => indices={indices}")

        return distances.reshape(-1), indices.reshape(-1)

    @property
    def K(self):
        """
        Number of nearest neighbors to return
        """
        return self._K

    @property
    def model(self):
        """
        Query model
        """
        return self._model
