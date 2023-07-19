from backend.feature_extractor.extractor import FeatureExtractor
from backend.ranker.ranker import Ranker
from PIL import Image
from io import BytesIO
import requests
from urllib.request import urlopen
import pytest
import numpy as np
import pickle

FEATURE_EXTRACTOR_PATH = './backend/ML-models/feature-extractor.pth'
RANKER_PATH = './backend/ML-models/ranker.pkl'
SCALER_PATH = './backend/ML-models/scaler.pkl'
TEST_IMAGE_URL = 'http://assets.myntassets.com/v1/images/style/\
properties/504a27acee8e6d89d7eec2fae5b5ef01_images.jpg'


# test feature extractor
@pytest.fixture
def extractor():
    return FeatureExtractor(model_path=FEATURE_EXTRACTOR_PATH,
                            device='cpu')


def test_feature_extractor(extractor: FeatureExtractor):
    url = TEST_IMAGE_URL
    image = Image.open(urlopen(url))
    assert extractor.extract(image).shape == extractor.output_shape


def test_feature_extractor_with_scaler(extractor: FeatureExtractor):
    url = TEST_IMAGE_URL
    image = Image.open(urlopen(url))

    with open(SCALER_PATH, 'rb') as scaler:
        extractor.set_scaler(pickle.load(scaler))

    assert extractor.extract(image).shape == extractor.output_shape


# test ranker
@pytest.fixture
def ranker():
    return Ranker(model_path=RANKER_PATH)


def test_ranker(ranker, extractor):
    query = np.random.rand(1, extractor.output_shape[0])
    assert ranker.rank(query).shape == (5,)


# test backend
def test_backend():
    image = Image.open(urlopen(TEST_IMAGE_URL))
    image = BytesIO(image.tobytes())

    # [TODO] change this to the actual url in docker network
    response = requests.post('http://localhost:8888/api/v1.0/predict',
                             files={'image': image})

    assert response.status_code == 200
