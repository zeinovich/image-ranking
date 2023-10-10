from ..feature_extractor.extractor import FeatureExtractor
from ..ranker.ranker import Ranker

from PIL import Image
from urllib.request import urlopen
import pytest
import numpy as np
import pickle
from dotenv import load_dotenv
from os import getenv

load_dotenv("backend.test.env")

FEATURE_EXTRACTOR_PATH = None
RANKER_PATH = getenv("RANKER_PATH")
SCALER_PATH = getenv("SCALER_PATH")
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons\
/thumb/b/b6/Image_created_with_a_mobile_phone.png\
/330px-Image_created_with_a_mobile_phone.png"


# test feature extractor
@pytest.fixture
def extractor():
    return FeatureExtractor(model_path=FEATURE_EXTRACTOR_PATH, device="cpu")


def test_feature_extractor(extractor: FeatureExtractor):
    url = TEST_IMAGE_URL
    image = Image.open(urlopen(url))
    assert extractor.transform is not None
    assert extractor.extract(image).shape == extractor.output_shape


def test_feature_extractor_with_scaler(extractor: FeatureExtractor):
    url = TEST_IMAGE_URL
    image = Image.open(urlopen(url))

    with open(SCALER_PATH, "rb") as scaler:
        extractor._scaler = pickle.load(scaler)

    assert extractor.extract(image).shape == extractor.output_shape


def test_feature_extractor_call(extractor: FeatureExtractor):
    url = TEST_IMAGE_URL
    image = Image.open(urlopen(url))
    assert extractor(image).shape == extractor.output_shape


def test_feature_extractor_call_with_scaler(extractor: FeatureExtractor):
    url = TEST_IMAGE_URL
    image = Image.open(urlopen(url))

    with open(SCALER_PATH, "rb") as scaler:
        extractor._scaler = pickle.load(scaler)

    assert extractor(image).shape == extractor.output_shape
    assert extractor.scaler is not None


def dummy_scaler():
    return None


def test_feature_extractor_wrong_scaler(extractor: FeatureExtractor):
    scaler = dummy_scaler()
    extractor._scaler = scaler

    assert extractor.scaler is None


def test_feature_extractor_repr(extractor: FeatureExtractor):
    assert str(extractor) is not None


# test ranker
@pytest.fixture
def ranker():
    return Ranker(RANKER_PATH)


def test_ranker(ranker: Ranker, extractor):
    query = np.random.rand(1, extractor.output_shape[0])
    assert ranker.rank(query).shape == (5,)
