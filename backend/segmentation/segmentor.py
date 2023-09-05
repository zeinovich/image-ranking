import cv2
import numpy as np
import torch
from torchvision import transforms

class DeepLabSegmentor:
    """
    Class for segmenting images
    """
    
    def __init__(self,
                 device: str = "cpu"):
        """
        device: device to be used for the model
        """
        self._device = device
        self._model = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'deeplabv3_resnet50', 
                                     pretrained=True)
        self._model.eval()
        self._model.to(self._device)
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    def _make_transparent_foreground(self, pic, mask):
        b, g, r = cv2.split(np.array(pic).astype('uint8'))
        a = np.ones(mask.shape, dtype='uint8') * 255
        alpha_im = cv2.merge([b, g, r, a], 4)
        bg = np.zeros(alpha_im.shape)
        new_mask = np.stack([mask, mask, mask, mask], axis=2)
        foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

        return foreground
    
    def _remove_background(self, input_img):
        preproccessed = self._transform(input_img)
        input_batch = preproccessed.unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(input_batch)['out'][0]

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
        