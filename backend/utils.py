import pandas as pd

from io import BytesIO
from PIL import Image
import base64
from sqlalchemy import Engine

from backend.feature_extractor import FeatureExtractor
from backend.ranker import Ranker


def load_image_from_json(data: dict) -> Image:
    """
    Load image from JSON

    Args:
        data (dict): JSON data

    Returns:
        Image: Image"""

    image = data["image"]
    image = Image.open(BytesIO(base64.b64decode(image)))
    return image


def get_predictions(
    image: Image, feature_extractor: FeatureExtractor, ranker: Ranker
) -> list:
    """
    Get predictions from image

    Args:
        image (Image): Image
        feature_extractor (FeatureExtractor): Feature extractor
        ranker (Ranker): Ranker

    Returns:
        list: Predictions"""

    features = feature_extractor.extract(image)
    ids = ranker.rank(features)

    return ids.tolist()


def get_info_from_db(ids: list, engine: Engine) -> dict:
    """
    Get info from DB

    Args:
        ids (list): Ids
        engine: SQLAlchemy Engine

    Returns:
        dict: Predictions"""

    # turn into tuple for SQL query
    ids = tuple(ids)

    query = f"""
    SELECT *
    FROM styles_v1 as t
    WHERE index IN {ids}
    """

    df = pd.read_sql(query, engine)
    predictions = df.to_dict(orient="records")

    return predictions


def dump_image(image: Image) -> str:
    """
    Dump image to base64 format to make it JSON serializable

    Args:
        image (Image): Image

    Returns:
        str: Image in base64 format"""

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    im_bytes = base64.b64encode(img_byte_arr).decode()
    return im_bytes
