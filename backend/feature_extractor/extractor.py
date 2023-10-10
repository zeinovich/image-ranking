import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop,
)
from torchvision.models import efficientnet_v2_s

from sklearn.preprocessing import StandardScaler
import numpy as np


class FeatureExtractor:
    """
    FeatureExtractor class

    Parameters
    ----------
    model_path : str
        Path to the model
    scaler : object, optional
        Scaler to be used to scale the output of the model, by default None
    device : str, optional
        Device to be used for the model, by default "cpu"

    Attributes
    ----------
    model : torch.nn.Module
        Query model
    device : str
        Device to be used for the model
    transform : torchvision.transforms.Compose
        Transform to be used for the model
    output_shape : tuple
        Output shape of the model

    Methods
    -------
    extract(image) -> np.ndarray
        Extract features from the image
    set_scaler(scaler) -> None
        Set the scaler to be used for scaling the output of the model

    Examples
    --------
    >>> from extractor import FeatureExtractor
    >>> extractor = FeatureExtractor(model_path="model.pth")
    >>> image = Image.open("image.jpg")
    >>> features = extractor.extract(image)
    >>> features.shape
    (OUTPUT_SHAPE,)
    """

    def __init__(
        self, model_path: str, scaler: StandardScaler = None, device: str = "cpu"
    ):

        """
        model_path: path to the model
        scaler: scaler to be used to scale the output of the model
        device: device to be used for the model
        """

        self._device = device
        self._model = efficientnet_v2_s()
        self._out_shape = 1280
        self._in_shape = 384
        self._model_name = self._model.__class__.__name__
        self._initialized = False

        if model_path is not None:
            self._model.load_state_dict(
                torch.load(model_path, map_location=torch.device(device))
            )
            self._initialized = True

        self._model = nn.Sequential(self._model.features, self._model.avgpool)

        self._transform = Compose(
            [
                ToTensor(),
                Resize((self._in_shape, self._in_shape), antialias=True),
                CenterCrop((self._in_shape, self._in_shape)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._scaler = scaler
        self._model = self._model.eval()

    @torch.no_grad()
    def extract(self, image) -> np.ndarray:

        """
        Extract features from the image
        image: image to be extracted

        returns: features of the image
        """

        image = self._transform(image).unsqueeze(0).to(self._device)

        out = self._model(image).cpu().numpy().flatten().reshape(1, -1)

        if self._scaler is not None:
            out = self._scaler.transform(out)

        return out.reshape(-1)

    @property
    def scaler(self) -> object:
        """
        Get the scaler
        """

        return self._scaler

    @property
    def model(self) -> torch.nn.Module:
        """
        Get the model
        """

        return self._model

    @property
    def device(self) -> str:
        """
        Get the device
        """

        return self._device

    @property
    def transform(self) -> Compose:
        """
        Get transform
        """

        return self._transform

    @property
    def output_shape(self) -> tuple:
        """
        Get output shape of the model
        """

        return (self._out_shape,)

    @property
    def input_shape(self) -> tuple:
        """
        Get input shape of the model
        """
        return (self._in_shape, self._in_shape)

    def __call__(self, image):
        return self.extract(image)

    def __repr__(self):
        return f"FeatureExtractor(model={self._model_name}, \
input_shape={self.input_shape}, \
output_shape={self.output_shape})"

    def __str__(self):
        return self.__repr__()
