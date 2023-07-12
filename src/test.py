from backend.feature_extractor.extractor import FeatureExtractor
from backend.ranker.ranker import Ranker
from PIL import Image
from urllib.request import urlopen
import pytest
import numpy as np

# test feature extractor
@pytest.fixture
def extractor():
    return FeatureExtractor(model_path='backend/ML-models/feature-extractor.pth',
                            device='cpu')


def test_feature_extractor(extractor):
    url = 'http://assets.myntassets.com/v1/images/style/properties/504a27acee8e6d89d7eec2fae5b5ef01_images.jpg'
    image = Image.open(urlopen(url))
    assert extractor.extract(image).shape == extractor.output_shape


# test ranker

@pytest.fixture
def ranker():
    return Ranker(model_path='backend/ML-models/ranker.pkl')


def test_ranker(ranker, extractor):
    query = np.random.rand(1, extractor.output_shape[0])
    assert ranker.rank(query).shape == (5,)