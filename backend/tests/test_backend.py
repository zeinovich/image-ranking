from backend.backend import (
    app,
    get_predictions,
    load_image_from_json,
    predict,
    get_info_from_db,
    health,
)
from backend.feature_extractor.extractor import FeatureExtractor
from backend.ranker.ranker import Ranker

from PIL import Image
from urllib.request import urlopen
import base64
from dotenv import load_dotenv
from os import getenv
import pytest

load_dotenv("./backend.env")

TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons\
/thumb/b/b6/Image_created_with_a_mobile_phone.png\
/330px-Image_created_with_a_mobile_phone.png"

FEATURE_EXTRACTOR_PATH = getenv("FEATURE_EXTRACTOR_PATH")
RANKER_PATH = getenv("RANKER_PATH")
SCALER_PATH = getenv("SCALER_PATH")


@pytest.fixture
def extractor():
    return FeatureExtractor(model_path=FEATURE_EXTRACTOR_PATH, device="cpu")


@pytest.fixture
def ranker():
    return Ranker(RANKER_PATH)


def test_load_image_from_json():
    # prepare image
    url = TEST_IMAGE_URL
    img_file = urlopen(url)
    im_bytes = img_file.read()
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    # prepare json
    data = {"image": im_b64}

    # load image
    image = load_image_from_json(data)

    assert isinstance(image, Image.Image)


def test_get_prediction(extractor: FeatureExtractor, ranker: Ranker):
    # prepare image
    url = TEST_IMAGE_URL
    img_file = urlopen(url)
    im_bytes = img_file.read()
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    # prepare json
    data = {"image": im_b64}

    # load image
    image = load_image_from_json(data)

    # get predictions
    predictions = get_predictions(image, extractor, ranker)

    assert isinstance(predictions, tuple)
    assert len(predictions) == 2
    assert isinstance(predictions[0], list)
    assert isinstance(predictions[1], list)
    assert len(predictions[0]) == len(predictions[1]) == ranker.K


def test_predict():
    with app.app_context():
        response = predict()
        assert response.status_code == 200


def test_get_info_from_db():
    DISTANCES = [123, 456, 789, 101112, 131415]
    IDS = [1, 2, 3, 4, 5]

    response = get_info_from_db(DISTANCES, IDS)
    assert response is None


def test_health():
    with app.app_context():
        response = health()
        assert response.status == "200 OK"
