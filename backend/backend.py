"""
Flask app for serving predictions

Usage:
    python backend.py
    gunicorn backend:app
"""

from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from PIL import Image

from io import BytesIO
import pandas as pd
import pickle
import logging
import base64
import traceback
import os

from feature_extractor import FeatureExtractor
from ranker import Ranker
from segmentation import Segmentor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("backend")
logger.info("[STARTED]")

FEATURE_EXTRACTOR_PATH = os.getenv("FEATURE_EXTRACTOR_PATH")
SEGMENTATION_MODEL = os.getenv("SEGMENTATION_MODEL")
RANKER_PATH = os.getenv("RANKER_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")

USER = os.getenv("POSTGRES_USER")
PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB = os.getenv("POSTGRES_DB")

logger.info(os.getcwd())
logger.info(f"{FEATURE_EXTRACTOR_PATH=}")
logger.info(f"{SEGMENTATION_MODEL=}")
logger.info(f"{RANKER_PATH=}")
logger.info(f"{SCALER_PATH=}")

logger.info(f"{USER=}")
logger.info(f"{PASSWORD=}")
logger.info(f"{DB=}")

logger.info("Successfully loaded environment")

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    logger.error(f"Error loading scaler: {e}\n{traceback.format_exc()}")
    logger.info("Setting scaler to None")
    scaler = None

try:
    feature_extractor = FeatureExtractor(
        model_path=FEATURE_EXTRACTOR_PATH, scaler=scaler
    )
    logger.info(f"{feature_extractor}")

    ranker = Ranker(model_path=RANKER_PATH)
    logger.info(f"{ranker}")

    segmentor = Segmentor(model_path=SEGMENTATION_MODEL)
    logger.info(f"{segmentor}")
    logger.info("Succesfully loaded models")

except Exception as e:
    logger.error(f"Error loading models: {e}\n{traceback.format_exc()}")


try:
    connection = f"postgresql://{USER}:{PASSWORD}@postgres:5432/{DB}"
    engine = create_engine(connection)
    logger.info("Connected to DB")

except Exception as e:
    logger.error(f"Error connecting to DB: {e}\n{traceback.format_exc()}")

app = Flask(__name__)
logger.info("[READY] Flask app ready")


def load_image_from_json(data: dict) -> Image:
    """
    Load image from JSON

    Args:
        data (dict): JSON data

    Returns:
        Image: Image"""

    image = data["image"]
    logger.info(f"Got image: {type(image)}")

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

    logger.info(f"Opened image: {type(image)} of size: {image.size}")
    features = feature_extractor.extract(image)
    logger.info(f"Extracted features: {features.shape}")

    ids = ranker.rank(features)
    logger.info(f"Predictions: {ids.shape}")
    return ids.tolist()


def get_info_from_db(ids: list) -> dict:
    """
    Get info from DB

    Args:
        distances (list): Distances
        ids (list): Ids

    Returns:
        dict: Predictions"""

    # turn into tuple for SQL query
    ids = tuple(ids)
    logger.info(f"Got ids: {ids}")

    query = f"""
    SELECT *
    FROM styles_v1 as t
    WHERE index IN {ids}
    """

    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Retrieved candidates: {df.shape}")

        predictions = df.to_dict(orient="records")
        logger.info(f'Candidates: {[pred["index"] for pred in predictions]}')

        return predictions

    except Exception as e:
        logger.error(f"Error getting info from DB: {e}\n{traceback.format_exc()}")


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


@app.route("/api/v1.0/predict", methods=["POST"])
def predict():

    logger.info("Got request")

    try:
        data = request.get_json()
        logger.info(f"Got JSON: {type(data)}")

        image = load_image_from_json(data)
        segmented_image, _ = segmentor.segment(image)

        # convert from RGBA to RGB and dump to base64
        segmented_image = Image.fromarray(segmented_image).convert("RGB")
        segmented_bytes = dump_image(segmented_image)

        ids = get_predictions(segmented_image, feature_extractor, ranker)
        predictions_urls = get_info_from_db(ids)

        if predictions_urls is None:
            return jsonify({"error": "No predictions found", "status_code": 404})

        return jsonify(
            {
                "predictions": predictions_urls,
                "segmented_image": segmented_bytes,
                "status_code": 200,
            }
        )

    except Exception as e:
        logger.error(f"Error in predict: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"{e}", "status_code": 400})


@app.route("/api/v1.0/health", methods=["GET"])
def health():
    return jsonify({"status": 200})


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8888)
