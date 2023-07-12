from feature_extractor.extractor import FeatureExtractor
from PIL import Image
from urllib.request import urlopen
import unittest
from unittest import TestCase

class TestFeatureExtractor(TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.extractor = FeatureExtractor(model_path='models:/FeatureExtractor/Staging',
                                          device=self.device)
        self.output_shape = self.extractor._get_output_shape()

    def test_extract(self):
        req = urlopen('http://assets.myntassets.com/v1/images/style/properties/504a27acee8e6d89d7eec2fae5b5ef01_images.jpg')
        image = Image.open(req)
        feature = self.extractor.extract(image)
        self.assertEqual(feature.shape[0], self.output_shape[1])

if __name__ == '__main__':
    unittest.main()

# command to run the test
# python -m unittest tests.fe_test.TestFeatureExtractor.test_extract