import cv2
import numpy as np
import torch
from torchvision import transforms


class Segmentor:
    """
    Model for segmenting images

    Wrapper above nn.Module

    ---
    Parameters
    ---
    model_path : str
        Path to the model
    device : str, optional
        Device to be used for the model, by default "cpu"

    ---
    Attributes
    ---
    model : torch.nn.Module
        Segmentation model
    device : str
        Device to be used for the model
    transform : torchvision.transforms.Compose
        Transform applied to the input image

    ---
    Methods
    ---
    segment(input_img) -> np.ndarray
        Segment the image
    ---
    Examples
    ---
    >>> from segmentor import Segmentor
    >>> segmentor = Segmentor(model_path="model.pth")
    >>> image = Image.open("image.jpg")
    >>> segmented_image, mask = segmentor.segment(image)
    >>> segmented_image.shape
    (H, W, 4)
    >>> mask.shape
    (H, W)
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self._device = device
        self._model = torch.load(model_path).eval()
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def model(self):
        """Returns the segmentation model"""
        return self._model

    @property
    def device(self):
        """Returns the device model is on"""
        return self._device

    @property
    def transform(self):
        """Returns the transform applied to the input image"""
        return self._transform

    def _make_transparent_foreground(self, pic, mask):
        """
        Make the foreground transparent

        `See Google Colab:
        https://colab.research.google.com/drive/1P9Pyq92ywLa6SRPmfnctgzScsXvJrrOM?usp=sharing#scrollTo=w_7SNhWQIZn7`

        pic: image to be segmented
        mask: mask of the image

        returns: image with transparent foreground"""

        b, g, r = cv2.split(np.array(pic).astype("uint8"))
        a = np.ones(mask.shape, dtype="uint8") * 255
        alpha_im = cv2.merge([b, g, r, a], 4)
        bg = np.full(alpha_im.shape, 255)
        new_mask = np.stack([mask, mask, mask, mask], axis=2)
        foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

        return foreground

    def _remove_background(self, input_img):
        """
        Remove the background from the image
        See Google Colab:
        `https://colab.research.google.com/drive/1P9Pyq92ywLa6SRPmfnctgzScsXvJrrOM?usp=sharing#scrollTo=w_7SNhWQIZn7`

        input_img: image to be segmented

        returns: segmented image
        """

        preproccessed = self._transform(input_img)
        input_batch = preproccessed.unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(input_batch)["out"][0]

        output_predictions = output.argmax(0)

        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        bin_mask = np.where(mask, 255, background).astype(np.uint8)
        print(bin_mask.shape)
        print(input_img)

        foreground = self._make_transparent_foreground(input_img, bin_mask)

        return foreground, bin_mask

    def segment(self, input_img):
        """
        Segment the image
        input_img: image to be segmented

        returns: segmented image
        """
        return self._remove_background(input_img)

    def __repr__(self) -> str:
        return "Segmentor(model=DeepLabV3_ResNet50)"
