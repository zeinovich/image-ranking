import pickle
import numpy as np


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
        Query model. Must have a `kneighbors` method
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

    def _candidates(self, query_point: np.ndarray) -> np.ndarray:
        return self._model.kneighbors(query_point, n_neighbors=self.K)[1]

    def rank(self, query: np.ndarray) -> np.ndarray:
        query_point = query.reshape(1, -1)
        indices = self._candidates(query_point)

        return indices.reshape(-1)

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

    def __repr__(self) -> str:
        return f"Ranker(model={self.model}, K={self.K})"
