import torch
from torchvision.transforms import Compose, ToTensor
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("./logs/extractor.log"),
        logging.StreamHandler(),
    ],
)


logger = logging.getLogger("extractor")


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
        self, model_path: str, scaler: str = None, device: str = "cpu"
    ):

        """
        model_path: path to the model
        scaler: scaler to be used to scale the output of the model
        device: device to be used for the model
        """

        self._device = device
        self._model = torch.load(model_path, map_location=self._device)
        self._transform = Compose([ToTensor()])
        self._scaler = scaler
        self._model.eval()

        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Using device {self._device}")
        logger.info(f"Using scaler {self._scaler}")
        logger.info(f"Using transform {self._transform}")

    @torch.no_grad()
    def extract(self, image) -> np.ndarray:

        """
        Extract features from the image
        image: image to be extracted

        returns: features of the image
        """

        image = self._transform(image).unsqueeze(0).to(self._device)
        logger.info(f"Extracting features from image of shape {image.shape}")

        out = self._model(image).cpu().numpy().flatten().reshape(1, -1)
        logger.info(f"Extracted features of shape {out.shape}")

        if self._scaler is not None:
            out = self._scaler.transform(out)
            logger.info(f"Scaled features of shape {out.shape}")

        return out.reshape(-1)

    def _get_output_shape(self, image_dim=(1, 3, 100, 100)) -> tuple:

        """
        Get the output shape of the model
        """

        out = self._model(torch.rand(*(image_dim))).data
        out = out.cpu().numpy().flatten().reshape(-1)
        return out.shape

    def set_scaler(self, scaler) -> None:
        """
        Set the scaler to be used for scaling the output of the model
        """

        self._scaler = scaler

        # test the scaler
        try:
            test_input = np.random.rand(1, self.output_shape[0])
            test_output = self._scaler.transform(test_input)

            assert (
                test_input.shape == test_output.shape
            ), f"Scaler {self._scaler} is not working properly"

        except Exception as e:
            logger.exception(e)
            self._scaler = None
            logger.info("Scaler set to None")

        logger.info(f"Scaler set to {self._scaler}")

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
        Get the transform
        """

        return self._transform

    @property
    def output_shape(self) -> tuple:
        """
        Get the output shape of the model
        """

        return self._get_output_shape()

    def __call__(self, image):
        return self.extract(image)

    def __repr__(self):
        return f"FeatureExtractor(model={self.model}, device={self.device})"

    def __str__(self):
        return self.__repr__()
